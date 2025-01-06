import streamlit as st
import cv2
import pandas as pd
import supervision as sv
import tempfile
import numpy as np
import os
import time
import io
import imageio
from scipy.spatial import distance
from pathlib import Path
from dataclasses import dataclass
from streamlit import session_state
from tqdm import tqdm
from inference import get_model
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram
)
from sports.common.team import TeamClassifier
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
CONFIG = SoccerPitchConfiguration()


# Set your Roboflow API key
os.environ['ROBOFLOW_API_KEY'] = 'WRMXZ9qvDiM4yLJTcH02'
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
BALL_DETECTION_MODEL_ID = "football-ball-detection-rejhg/4"
PLAYER_DETECTION_MODEL_ID = "football_data-6lerg/2"
FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"

# Cache the model

@st.cache_resource
def load_model(model_id):
    return get_model(model_id=model_id, api_key=ROBOFLOW_API_KEY)

PLAYER_DETECTION_MODEL = load_model(PLAYER_DETECTION_MODEL_ID)
BALL_DETECTION_MODEL = load_model(BALL_DETECTION_MODEL_ID)
FIELD_DETECTION_MODEL = load_model(FIELD_DETECTION_MODEL_ID)

from typing import Optional

def draw_pitch_voronoi_diagram_2(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    # Increase steepness of the blend effect
    steepness = 15  # Increased steepness for sharper transition
    distance_ratio = min_distances_team_2 / np.clip(min_distances_team_1 + min_distances_team_2, a_min=1e-5, a_max=None)
    blend_factor = np.tanh((distance_ratio - 0.5) * steepness) * 0.5 + 0.5

    # Create the smooth color transition
    for c in range(3):  # Iterate over the B, G, R channels
        voronoi[:, :, c] = (blend_factor * team_1_color_bgr[c] +
                            (1 - blend_factor) * team_2_color_bgr[c]).astype(np.uint8)

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay

# Resolve_goalkeepers
def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)

class PassDetection:
    def __init__(self):
        self.team1_passes_success = 0
        self.team1_passes_fail = 0
        self.team2_passes_success = 0
        self.team2_passes_fail = 0
        self.ball_owner_tracker_id = None

    def detect_pass(self, player_detection, ball_detections, team1_players_tracker_ids, team2_players_tracker_ids):
        # Extract necessary data from player and ball detections
        player_tracker_id = player_detection[4]
        global xyxy
        global player_center
        xyxy = player_detection[0]
        player_center = np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])
        bottom_left = np.array([xyxy[0], xyxy[3]])
        bottom_right = np.array([xyxy[2], xyxy[3]])

        if ball_detections.xyxy.shape[0] > 0:
            ball_xyxy = ball_detections.xyxy[0]
            ball_center = np.array([(ball_xyxy[0] + ball_xyxy[2]) / 2, (ball_xyxy[1] + ball_xyxy[3]) / 2])

            # Calculate distance between player and ball
            distance = min(np.linalg.norm(bottom_left - ball_center), np.linalg.norm(bottom_right - ball_center))

            # Check if a pass has occurred based on the distance
            if 0 < distance <= 20:
                return self.handle_pass(player_tracker_id, player_detection, team1_players_tracker_ids, team2_players_tracker_ids)

        return None  # No pass detected

    def handle_pass(self, player_tracker_id, player_detection, team1_players_tracker_ids, team2_players_tracker_ids):
        # Update pass success/fail counters
        if self.ball_owner_tracker_id is None:
            self.ball_owner_tracker_id = player_tracker_id
        elif self.ball_owner_tracker_id != player_tracker_id:
            if player_detection[3] == 0:  # Team 1
                if self.ball_owner_tracker_id in team1_players_tracker_ids:
                    self.team1_passes_success += 1
                else:
                    self.team1_passes_fail += 1
                triangle_color = (0, 0, 0)
            else:  # Team 2
                if self.ball_owner_tracker_id in team2_players_tracker_ids:
                    self.team2_passes_success += 1
                else:
                    self.team2_passes_fail += 1
                triangle_color = (255, 255, 255)

            triangle_points = np.array([
              (player_center[0], xyxy[1] - 20),  # Top point above the player's head
              (player_center[0] - 10, xyxy[1] - 40),  # Left point
              (player_center[0] + 10, xyxy[1] - 40)   # Right point
            ])

            cv2.fillPoly(annotated_frame, [triangle_points.astype(np.int32)], triangle_color)  # Draw the triangle  # Draw triangle (gold color)

            # Update ball owner after pass
            self.ball_owner_tracker_id = player_tracker_id
            return True  # Pass detected
        return False  # No pass detected

    def calculate_possession_percentage(self):
        """
        Calculate possession for each team based on successful passes.
        Possession is calculated as the number of successful passes by each team,
        divided by the total number of passes, and converted to a percentage.
        """
        total_passes = self.team1_passes_success + self.team2_passes_success
        if total_passes == 0:  # Avoid division by zero if no passes are made
            return 0, 0

        team1_possession_percentage = (self.team1_passes_success / total_passes) * 100
        team2_possession_percentage = (self.team2_passes_success / total_passes) * 100

        return team1_possession_percentage, team2_possession_percentage

class AdvancedFootballAnalytics:
    def __init__(self):
        self.team1_heatmap = np.zeros((100, 100))  # Normalized coordinate space
        self.team2_heatmap = np.zeros((100, 100))
        self.ball_heatmap = np.zeros((100, 100))
        self.team_formations = {1: [], 2: []}
        self.ball_speed_data = []
        self.pressure_events = []

    def normalize_coordinates(self, x, y, frame_width, frame_height):
        """Normalize coordinates to 100x100 space regardless of video resolution"""
        norm_x = int((x / frame_width) * 100)
        norm_y = int((y / frame_height) * 100)
        return max(0, min(99, norm_x)), max(0, min(99, norm_y))

    def update_metrics(self, frame, all_detections, ball_detections, team1_ids, team2_ids, frame_width, frame_height):
        # Update heatmaps for players
        for i in range(len(all_detections)):
            # Get detection coordinates
            xyxy = all_detections.xyxy[i]
            tracker_id = all_detections.tracker_id[i]

            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            norm_x, norm_y = self.normalize_coordinates(x_center, y_center, frame_width, frame_height)

            if tracker_id in team1_ids:
                self.team1_heatmap[norm_y, norm_x] += 1
            elif tracker_id in team2_ids:
                self.team2_heatmap[norm_y, norm_x] += 1

        # Track ball speed and location
        if len(ball_detections.xyxy) > 0:
            ball_xyxy = ball_detections.xyxy[0]
            ball_x = (ball_xyxy[0] + ball_xyxy[2]) / 2
            ball_y = (ball_xyxy[1] + ball_xyxy[3]) / 2
            norm_ball_x, norm_ball_y = self.normalize_coordinates(ball_x, ball_y, frame_width, frame_height)
            self.ball_heatmap[norm_ball_y, norm_ball_x] += 1

            if len(self.ball_speed_data) > 0:
                prev_x, prev_y = self.ball_speed_data[-1][:2]  # Get only x,y from previous data
                speed = np.sqrt((norm_ball_x - prev_x)**2 + (norm_ball_y - prev_y)**2)
                self.ball_speed_data.append((norm_ball_x, norm_ball_y, speed))
            else:
                self.ball_speed_data.append((norm_ball_x, norm_ball_y, 0))

        # Analyze team formations
        team1_positions = []
        team2_positions = []

        for i in range(len(all_detections)):
            xyxy = all_detections.xyxy[i]
            tracker_id = all_detections.tracker_id[i]

            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            norm_x, norm_y = self.normalize_coordinates(x_center, y_center, frame_width, frame_height)

            if tracker_id in team1_ids:
                team1_positions.append((norm_x, norm_y))
            elif tracker_id in team2_ids:
                team2_positions.append((norm_x, norm_y))

        self.team_formations[1].append(team1_positions)
        self.team_formations[2].append(team2_positions)

        # Analyze pressure events
        for t1_pos in team1_positions:
            for t2_pos in team2_positions:
                dist = distance.euclidean(t1_pos, t2_pos)
                if dist < 5:  # Threshold in normalized coordinates
                    self.pressure_events.append((t1_pos, t2_pos))

    def get_analytics(self):
        """Calculate and return all analytics metrics"""
        analytics = {
            'team1_coverage': np.count_nonzero(self.team1_heatmap) / self.team1_heatmap.size * 100,
            'team2_coverage': np.count_nonzero(self.team2_heatmap) / self.team2_heatmap.size * 100,
            'ball_coverage': np.count_nonzero(self.ball_heatmap) / self.ball_heatmap.size * 100,
            'avg_ball_speed': np.mean([speed for _, _, speed in self.ball_speed_data[1:]]) if len(self.ball_speed_data) > 1 else 0,
            'max_ball_speed': np.max([speed for _, _, speed in self.ball_speed_data[1:]]) if len(self.ball_speed_data) > 1 else 0,
            'pressure_events_count': len(self.pressure_events),
            'team1_avg_width': self._calculate_team_width(1),
            'team2_avg_width': self._calculate_team_width(2),
            'team1_avg_depth': self._calculate_team_depth(1),
            'team2_avg_depth': self._calculate_team_depth(2)
        }
        return analytics

    def _calculate_team_width(self, team_id):
        formations = self.team_formations[team_id]
        if not formations:
            return 0
        widths = []
        for positions in formations:
            if positions:
                x_coords = [pos[0] for pos in positions]
                widths.append(max(x_coords) - min(x_coords))
        return np.mean(widths) if widths else 0

    def _calculate_team_depth(self, team_id):
        formations = self.team_formations[team_id]
        if not formations:
            return 0
        depths = []
        for positions in formations:
            if positions:
                y_coords = [pos[1] for pos in positions]
                depths.append(max(y_coords) - min(y_coords))
        return np.mean(depths) if depths else 0

def run_radar(SOURCE_VIDEO_PATH, confidence_threshold=0.3, st_progress_bar=None):
    # Initialize constants
    BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID = 0, 1, 2, 3

    # Load video and setup writer
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width, video_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    video_buffer = io.BytesIO()
    writer = imageio.get_writer(video_buffer, format='mp4', fps=30)

    # Initialize annotators and tracker
    ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), thickness=2)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                                        text_color=sv.Color.from_hex('#000000'), text_position=sv.Position.BOTTOM_CENTER)
    triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'), base=20, height=17)
    tracker = sv.ByteTrack()
    tracker.reset()

    analytics = AdvancedFootballAnalytics()

    # Initialize team classifier
    team_classifier = TeamClassifier(device="cuda")

    # Collect player crops for team classification
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH, stride=30)
    crops = []
    for frame in tqdm(frame_generator, desc='Collecting crops'):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=confidence_threshold)[0]
        detections = sv.Detections.from_inference(result)
        players_detections = detections[detections.class_id == PLAYER_ID]
        crops.extend([sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy])

    # Fit the team classifier
    team_classifier.fit(crops)

    # Initialize PassDetection for tracking passes
    pass_detection = PassDetection()

    # Initialize tracking variables
    frame_count = 0
    team1_players_tracker_ids, team2_players_tracker_ids = [], []

    # Initialize an empty list to store ball positions
    ball_positions = []

    # Max number of points to show the path (you can adjust this)
    MAX_PATH_POINTS = 50

    analytics_overlay = {
        'team1_possession': 0,
        'team2_possession': 0,
        'ball_speed': 0,
        'pressure_events': 0
    }

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect ball and players
        ball_result = BALL_DETECTION_MODEL.infer(frame, confidence=confidence_threshold)[0]
        ball_detections = sv.Detections.from_inference(ball_result)
        ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)

        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=confidence_threshold)[0]
        detections = sv.Detections.from_inference(result)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        # Separate detections by class
        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        # Assign teams to players
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)
        goalkeepers_team_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
        goalkeepers_detections.class_id = goalkeepers_team_id
        referees_detections.class_id -= 1  # Adjust referee class ID if needed

        all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

        # Update team tracker lists
        team1_players_tracker_ids.clear()
        team2_players_tracker_ids.clear()
        for player_detection in players_detections:
            if player_detection[3] == 0:
                team1_players_tracker_ids.append(player_detection[4])
            else:
                team2_players_tracker_ids.append(player_detection[4])

        analytics.update_metrics(
            frame,
            all_detections,
            ball_detections,
            team1_players_tracker_ids,
            team2_players_tracker_ids,
            video_width,
            video_height
        )

        # Create labels for tracked detections
        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
        all_detections.class_id = all_detections.class_id.astype(int)

        # Annotate frame
        global annotated_frame
        annotated_frame = frame.copy()

        if ball_detections.xyxy.shape[0] > 0:
            # Get the ball's center
            ball_xyxy = ball_detections.xyxy[0]
            ball_center = np.array([(ball_xyxy[0] + ball_xyxy[2]) / 2, (ball_xyxy[1] + ball_xyxy[3]) / 2])

            # Add the current ball position to the list
            ball_positions.append(ball_center)
            if len(ball_positions) > MAX_PATH_POINTS:
                ball_positions.pop(0)  # Remove the oldest position if we exceed the limit

        # Draw the path of the ball
        for i in range(1, len(ball_positions)):
            cv2.line(annotated_frame,
                     tuple(ball_positions[i-1].astype(int)),
                     tuple(ball_positions[i].astype(int)),
                      (255, 255, 255), 2)  # Green line

        # Detect passes using the PassDetection class
        for player_detection in players_detections:
            pass_detection.detect_pass(player_detection, ball_detections, team1_players_tracker_ids, team2_players_tracker_ids)

        current_analytics = analytics.get_analytics()
        analytics_overlay['ball_speed'] = current_analytics['avg_ball_speed']
        analytics_overlay['pressure_events'] = current_analytics['pressure_events_count']

        # Add analytics overlay to frame
        cv2.putText(annotated_frame,
                    f"Ball Speed: {analytics_overlay['ball_speed']:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame,
                    f"Pressure Events: {analytics_overlay['pressure_events']}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Annotate the frame with detection results
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
        annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)

        # Write frame to output video
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)
        frame_count += 1

        # Update progress bar
        if st_progress_bar:
            progress_value = min(frame_count / total_frames, 1.0)
            st_progress_bar.progress(progress_value)

    # Release resources
    cap.release()
    writer.close()

    final_analytics = analytics.get_analytics()
    team1_possession_percentage, team2_possession_percentage = pass_detection.calculate_possession_percentage()

    # Store results in session state
    video_buffer.seek(0)
    st.session_state.video_data = video_buffer.getvalue()

    st.session_state.team1_passes_success = pass_detection.team1_passes_success
    st.session_state.team1_passes_fail = pass_detection.team1_passes_fail
    st.session_state.team2_passes_success = pass_detection.team2_passes_success
    st.session_state.team2_passes_fail = pass_detection.team2_passes_fail
    st.session_state.team1_possession = team1_possession_percentage
    st.session_state.team2_possession = team2_possession_percentage

    st.session_state.advanced_analytics = final_analytics

    st.text("Basic Stats:")
    st.text(f"Team 1 Possession: {team1_possession_percentage:.1f}%")
    st.text(f"Team 2 Possession: {team2_possession_percentage:.1f}%")

    st.text("\nAdvanced Analytics:")
    st.text(f"Team 1 Field Coverage: {final_analytics['team1_coverage']:.1f}%")
    st.text(f"Team 2 Field Coverage: {final_analytics['team2_coverage']:.1f}%")
    st.text(f"Ball Coverage: {final_analytics['ball_coverage']:.1f}%")
    st.text(f"Average Ball Speed: {final_analytics['avg_ball_speed']:.1f}")
    st.text(f"Total Pressure Events: {final_analytics['pressure_events_count']}")
    st.text(f"Team 1 Avg Formation Width: {final_analytics['team1_avg_width']:.1f}")
    st.text(f"Team 2 Avg Formation Width: {final_analytics['team2_avg_width']:.1f}")

    return st.session_state.video_data, total_frames

def render_radar(SOURCE_VIDEO_PATH, confidence_threshold=0.3, st_progress_bar=None):
    # Initialize constants
    BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID = 0, 1, 2, 3
    global annotated_frame
    # Load video and setup writer
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width, video_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    video_buffer = io.BytesIO()
    writer = imageio.get_writer(video_buffer, format='mp4', fps=30)

    # Initialize annotators and tracker
    ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), thickness=2)
    label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                                        text_color=sv.Color.from_hex('#000000'), text_position=sv.Position.BOTTOM_CENTER)
    triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'), base=20, height=17)
    tracker = sv.ByteTrack()
    tracker.reset()

    # Initialize team classifier
    team_classifier = TeamClassifier(device="cuda")

    # Collect player crops for team classification
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH, stride=30)
    crops = []
    for frame in tqdm(frame_generator, desc='Collecting crops'):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=confidence_threshold)[0]
        detections = sv.Detections.from_inference(result)
        players_detections = detections[detections.class_id == PLAYER_ID]
        crops.extend([sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy])

    # Fit the team classifier
    team_classifier.fit(crops)

    # Initialize PassDetection for tracking passes
    pass_detection = PassDetection()


    # Initialize tracking variables
    frame_count = 0
    team1_players_tracker_ids, team2_players_tracker_ids = [], []

    # Initialize an empty list to store ball positions
    ball_positions = []

    # Max number of points to show the path (you can adjust this)
    MAX_PATH_POINTS = 50

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect ball and players
        ball_result = BALL_DETECTION_MODEL.infer(frame, confidence=confidence_threshold)[0]
        ball_detections = sv.Detections.from_inference(ball_result)
        ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)

        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=confidence_threshold)[0]
        detections = sv.Detections.from_inference(result)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        # Separate detections by class
        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        # Assign teams to players
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)
        goalkeepers_team_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
        goalkeepers_detections.class_id = goalkeepers_team_id
        referees_detections.class_id -= 1  # Adjust referee class ID if needed

        all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

        #Detect pitch key points
        result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)

        # Transform points from image coordinates to pitch coordinates
        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(CONFIG.vertices)[filter]
        transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

        # Project ball, players, and referees onto the pitch
        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

        players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_players_xy = transformer.transform_points(points=players_xy)

        referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_referees_xy = transformer.transform_points(points=referees_xy)


        # Update team tracker lists
        team1_players_tracker_ids.clear()
        team2_players_tracker_ids.clear()
        for player_detection in players_detections:
            if player_detection[3] == 0:
                team1_players_tracker_ids.append(player_detection[4])
            else:
                team2_players_tracker_ids.append(player_detection[4])

        # Create labels for tracked detections
        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
        all_detections.class_id = all_detections.class_id.astype(int)

        # Annotate frame

        annotated_frame = frame.copy()

        if ball_detections.xyxy.shape[0] > 0:
            # Get the ball's center
            ball_xyxy = ball_detections.xyxy[0]
            ball_center = np.array([(ball_xyxy[0] + ball_xyxy[2]) / 2, (ball_xyxy[1] + ball_xyxy[3]) / 2])

            # Add the current ball position to the list
            ball_positions.append(ball_center)
            if len(ball_positions) > MAX_PATH_POINTS:
                ball_positions.pop(0)  # Remove the oldest position if we exceed the limit

        # Draw the path of the ball
        for i in range(1, len(ball_positions)):
            cv2.line(annotated_frame, tuple(ball_positions[i-1].astype(int)), tuple(ball_positions[i].astype(int)), (255, 255, 255), 2)  # Green line

        # Detect passes using the PassDetection class
        for player_detection in players_detections:
            pass_detection.detect_pass(player_detection, ball_detections, team1_players_tracker_ids, team2_players_tracker_ids)

        # Annotate the frame with detection results
        annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
        annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)


        annotated_frame = draw_pitch(
            config=CONFIG,
            background_color=sv.Color.WHITE,
            line_color=sv.Color.BLACK
        )
        annotated_frame = draw_pitch_voronoi_diagram_2(
            config=CONFIG,
            team_1_xy=pitch_players_xy[players_detections.class_id == 0],
            team_2_xy=pitch_players_xy[players_detections.class_id == 1],
            team_1_color=sv.Color.from_hex('00BFFF'),
            team_2_color=sv.Color.from_hex('FF1493'),
            pitch=annotated_frame)
        annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.WHITE,
            radius=8,
            thickness=1,
            pitch=annotated_frame)
        annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 0],
            face_color=sv.Color.from_hex('00BFFF'),
            edge_color=sv.Color.WHITE,
            radius=16,
            thickness=1,
            pitch=annotated_frame)
        annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 1],
            face_color=sv.Color.from_hex('FF1493'),
            edge_color=sv.Color.WHITE,
            radius=16,
            thickness=1,
            pitch=annotated_frame)

        # Write frame to output video
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)
        frame_count += 1

        # Update progress bar
        if st_progress_bar:
            progress_value = min(frame_count / total_frames, 1.0)
            st_progress_bar.progress(progress_value)

    # Release resources
    cap.release()
    writer.close()

    team1_possession_percentage, team2_possession_percentage = pass_detection.calculate_possession_percentage()

    # Store results in session state
    video_buffer.seek(0)
    st.session_state.video_data = video_buffer.getvalue()

    st.session_state.team1_passes_success = pass_detection.team1_passes_success
    st.session_state.team1_passes_fail = pass_detection.team1_passes_fail
    st.session_state.team2_passes_success = pass_detection.team2_passes_success
    st.session_state.team2_passes_fail = pass_detection.team2_passes_fail

    st.session_state.team1_possession_percentage = team1_possession_percentage
    st.session_state.team2_possession_percentage = team2_possession_percentage

    st.text(team1_possession_percentage)
    st.text(team2_possession_percentage)
    return st.session_state.video_data, total_frames




def show_introduction():
    st.title("⚽ Ứng dụng nhận diện, theo dõi và phân tích Bóng đá")
    st.markdown("""
    Chào mừng đến với **Ứng dụng nhận diện, theo dõi và phân tích Bóng đá !**
    Ứng dụng này cho phép bạn tải lên một video trận đấu bóng đá và phát hiện các cầu thủ trong các khung hình bằng AI.

    **Các tính năng chính**:
    - Tải lên video bóng đá ở định dạng MP4, MOV hoặc AVI..
    - Phát hiện và chú thích các cầu thủ trong các khung hình.
    - Tải xuống video đã xử lý với chú thích cầu thủ.

    **Cách sử dụng**:
    1. Tải lên một video trận đấu bóng đá.
    2. Nhấp vào 'Phát hiện cầu thủ' để chạy mô hình AI.
    3. Tải xuống video đã xử lý với chú thích.
    """)

def show_player_detection_page():
    st.title("🎥 Khởi chạy mô hình nhận diện")

    # Initialize session state variables
    if 'uploaded_video' not in st.session_state:
        st.session_state.uploaded_video = None
    if 'video_data' not in st.session_state:
        st.session_state.video_data = None
    if 'total_frames' not in st.session_state:
        st.session_state.total_frames = 0

    # Sidebar for settings
    with st.sidebar:
        st.header("Cài đặt nhận diện")
        confidence_threshold = st.slider(
            "Ngưỡng tự tin",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Điều chỉnh ngưỡng tự tin của mô hình"
        )

    # Upload video
    uploaded_video = st.file_uploader(
        "Tải lên video bóng đá của bạn",
        type=["mp4", "mov", "avi"],
        help="Tải lên video để phân tích"
    )

    if uploaded_video is not None:
        st.session_state.uploaded_video = uploaded_video
        video_bytes = uploaded_video.read()
        st.video(video_bytes)

        # Temporary directory setup
        temp_dir = tempfile.mkdtemp()
        temp_input_path = Path(temp_dir) / "input_video.mp4"
        with open(temp_input_path, "wb") as f:
            f.write(video_bytes)

        # Extract video information
        cap = cv2.VideoCapture(str(temp_input_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps
        cap.release()

        st.session_state.total_frames = total_frames

        # Display video info
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng số khung hình", f"{total_frames:,}")
        col2.metric("FPS", f"{fps}")
        col3.metric("Thời gian", f"{duration:.2f}s")

        # Start Detection
        if st.button("Bắt đầu nhận diện", key="start_detection"):
            progress_bar = st.progress(0)
            try:
                # Run run_radar function
                video_data, _ = run_radar(
                    SOURCE_VIDEO_PATH=str(temp_input_path),
                    confidence_threshold=confidence_threshold,
                    st_progress_bar=progress_bar
                )

                st.session_state.video_data = video_data

                # Display selected frames
                st.subheader("Kết quả nhận diện")

                # Video playback
                st.subheader("Video đã qua xử lý")
                st.video(video_data, format="video/mp4")

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                progress_bar.empty()

            # Cleanup temporary files
            try:
                os.remove(temp_input_path)
                os.rmdir(temp_dir)
            except Exception as cleanup_error:
                st.warning(f"Cleanup Error: {str(cleanup_error)}")

        # Run ren_radar Button
        if st.button("Chạy nhận diện với video radar", key="ren_detection"):
            progress_bar = st.progress(0)
            try:
                # Run ren_radar function
                video_data, _ = render_radar(
                    SOURCE_VIDEO_PATH=str(temp_input_path),
                    confidence_threshold=confidence_threshold,
                    st_progress_bar=progress_bar
                )

                st.session_state.video_data = video_data

                # Display selected frames
                st.subheader("Kết quả nhận diện")

                # Video playback
                st.subheader("Video đã xử lý")
                st.video(video_data, format="video/mp4")

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                progress_bar.empty()

            # Cleanup temporary files
            try:
                os.remove(temp_input_path)
                os.rmdir(temp_dir)
            except Exception as cleanup_error:
                st.warning(f"Cleanup Error: {str(cleanup_error)}")

    # How-to section
    with st.expander("Hướng dẫn cách xử dụng"):
        st.markdown("""
        1. Tải lên một video để phân tích.
        2. Điều chỉnh ngưỡng tự tin nếu cần thiết.
        3. Bắt đầu nhận diện và xem kết quả.
        4. Tải về video kết quả.
        """)

def show_maps():
    st.title("📊 Biểu đồ")
    st.markdown("""Các thông số hiện không khả dụng, hãy chạy mô hình nhận diện trước.""")

def show_page_4():
    # Add custom CSS styling
    st.markdown("""
        <style>
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
            margin-bottom: 1rem;
        }
        .chart-container {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header-container {
            padding: 1rem 0;
            margin-bottom: 1rem;
            border-bottom: 2px solid #dee2e6;
        }
        .explanation {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>📊 Bảng Phân Tích Trận Đấu</h1>", unsafe_allow_html=True)

    # Check for required data
    required_keys = ['team1_passes_success', 'team1_possession', 'advanced_analytics']
    if not all(key in st.session_state for key in required_keys):
        st.error("🚫 Dữ liệu không khả dụng. Vui lòng chạy mô hình nhận diện trước.")
        return

    # Get data from session state
    team1_success = st.session_state.team1_passes_success
    team1_fail = st.session_state.team1_passes_fail
    team2_success = st.session_state.team2_passes_success
    team2_fail = st.session_state.team2_passes_fail
    team1_possession = st.session_state.team1_possession
    team2_possession = st.session_state.team2_possession
    analytics = st.session_state.advanced_analytics

    # Calculate basic metrics
    team1_total = team1_success + team1_fail
    team2_total = team2_success + team2_fail
    team1_success_rate = (team1_success / team1_total * 100) if team1_total > 0 else 0
    team2_success_rate = (team2_success / team2_total * 100) if team2_total > 0 else 0

    # Create main tabs
    tab1, tab2, tab3 = st.tabs(["📌 Tổng Quan", "📊 Thống Kê Chi Tiết", "📈 Phân Tích Trực Quan"])

    # Tab 1: Overview
    with tab1:
        st.markdown("<div class='header-container'><h2>Tổng Quan Trận Đấu</h2></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        metrics = [
            (col1, "Kiểm Soát Bóng", f"{team1_possession:.1f}% vs {team2_possession:.1f}%",
             f"{team1_possession - team2_possession:.1f}% chênh lệch"),
            (col2, "Đường Chuyền Thành Công", f"{team1_success} vs {team2_success}",
             f"{team1_success - team2_success} chênh lệch"),
            (col3, "Tỷ Lệ Chuyền Thành Công", f"{team1_success_rate:.1f}% vs {team2_success_rate:.1f}%",
             f"{team1_success_rate - team2_success_rate:.1f}% chênh lệch")
        ]

        for col, label, value, delta in metrics:
            with col:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric(label, value, delta)
                st.markdown("</div>", unsafe_allow_html=True)

        # Expanders for metric explanations
        with st.expander("ℹ️ Giải thích các chỉ số"):
            st.markdown("""
                ### 🎯 Kiểm Soát Bóng
                - Thể hiện phần trăm thời gian mỗi đội kiểm soát bóng
                - Càng cao càng thể hiện khả năng kiểm soát trận đấu tốt

                ### 🎯 Đường Chuyền Thành Công
                - Số lượng đường chuyền chính xác của mỗi đội
                - Phản ánh khả năng phối hợp và chất lượng chuyền bóng

                ### 🎯 Tỷ Lệ Chuyền Thành Công
                - Phần trăm đường chuyền thành công trên tổng số đường chuyền
                - Chỉ số quan trọng đánh giá độ chính xác trong chuyền bóng
            """)

    # Tab 2: Detailed Stats
    with tab2:
        st.markdown("<div class='header-container'><h2>Phân Tích Nâng Cao</h2></div>", unsafe_allow_html=True)

        # Formation Analysis
        st.subheader("📐 Phân Tích Đội Hình")
        form_col1, form_col2 = st.columns(2)

        for col, team_num, width, depth in [
            (form_col1, 1, analytics['team1_avg_width'], analytics['team1_avg_depth']),
            (form_col2, 2, analytics['team2_avg_width'], analytics['team2_avg_depth'])
        ]:
            with col:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.markdown(f"##### Đội {team_num}")
                st.metric("Độ Rộng Đội Hình", f"{width:.1f}")
                st.metric("Độ Sâu Đội Hình", f"{depth:.1f}")
                st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("ℹ️ Giải thích về phân tích đội hình"):
            st.markdown("""
                ### 📌 Độ Rộng Đội Hình
                - Thể hiện khoảng cách trung bình giữa các cầu thủ theo chiều ngang
                - Đội hình rộng giúp tạo không gian và khai thác biên

                ### 📌 Độ Sâu Đội Hình
                - Thể hiện khoảng cách trung bình giữa các cầu thủ theo chiều dọc
                - Ảnh hưởng đến khả năng phản công và phòng ngự
            """)

        # Ball Analysis
        st.subheader("⚽ Phân Tích Chuyển Động Bóng")
        ball_col1, ball_col2 = st.columns(2)

        with ball_col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Tốc Độ Bóng Trung Bình", f"{analytics['avg_ball_speed']:.1f}")
            st.metric("Tốc Độ Bóng Tối Đa", f"{analytics['max_ball_speed']:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with ball_col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric("Số Lần Tạo Áp Lực", str(analytics['pressure_events_count']))
            st.metric("Vùng Di Chuyển Bóng", f"{analytics['ball_coverage']:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("ℹ️ Giải thích về phân tích chuyển động bóng"):
            st.markdown("""
                ### 🏃 Tốc Độ Bóng
                - **Trung bình**: Tốc độ di chuyển thông thường của bóng trong trận
                - **Tối đa**: Tốc độ cao nhất đạt được (thường trong các pha sút hoặc chuyền dài)

                ### 🎯 Các Chỉ Số Khác
                - **Số Lần Tạo Áp Lực**: Số lần cầu thủ tạo áp lực lên đối thủ
                - **Vùng Di Chuyển Bóng**: Phần trăm diện tích sân mà bóng đã di chuyển qua
            """)

    # Tab 3: Visual Analysis
    with tab3:
        st.markdown("<div class='header-container'><h2>Phân Tích Trực Quan</h2></div>", unsafe_allow_html=True)

        # Visualization controls
        viz_col1, viz_col2 = st.columns(2)
        viz_type = viz_col1.selectbox(
            "Chọn Loại Phân Tích",
            ["Thống Kê Chuyền Bóng", "Phân Tích Kiểm Soát Bóng", "Vùng Hoạt Động", "Phân Tích Đội Hình"]
        )
        chart_type = viz_col2.selectbox(
            "Chọn Loại Biểu Đồ",
            ["Biểu Đồ Cột", "Biểu Đồ Đường", "Biểu Đồ Vùng"]
        )

        # Prepare chart data
        if viz_type == "Thống Kê Chuyền Bóng":
            chart_data = pd.DataFrame({
                'Loại': ['Thành Công', 'Thất Bại'],
                'Đội 1': [team1_success, team1_fail],
                'Đội 2': [team2_success, team2_fail]
            })
        elif viz_type == "Phân Tích Kiểm Soát Bóng":
            chart_data = pd.DataFrame({
                'Loại': ['Kiểm Soát'],
                'Đội 1': [team1_possession],
                'Đội 2': [team2_possession]
            })
        elif viz_type == "Vùng Hoạt Động":
            chart_data = pd.DataFrame({
                'Loại': ['Vùng Hoạt Động'],
                'Đội 1': [analytics['team1_coverage']],
                'Đội 2': [analytics['team2_coverage']],
                'Bóng': [analytics['ball_coverage']]
            })
        else:  # Formation Analysis
            chart_data = pd.DataFrame({
                'Loại': ['Độ Rộng', 'Độ Sâu'],
                'Đội 1': [analytics['team1_avg_width'], analytics['team1_avg_depth']],
                'Đội 2': [analytics['team2_avg_width'], analytics['team2_avg_depth']]
            })

        # Display chart
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        if chart_type == "Biểu Đồ Cột":
            st.bar_chart(chart_data.set_index('Loại'))
        elif chart_type == "Biểu Đồ Đường":
            st.line_chart(chart_data.set_index('Loại'))
        else:  # Area Chart
            st.area_chart(chart_data.set_index('Loại'))
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("ℹ️ Hướng dẫn đọc biểu đồ"):
            st.markdown("""
                ### 📊 Cách Đọc Các Biểu Đồ

                #### Thống Kê Chuyền Bóng
                - So sánh số lượng đường chuyền thành công và thất bại giữa hai đội
                - Càng cao càng thể hiện khả năng kiểm soát bóng tốt

                #### Phân Tích Kiểm Soát Bóng
                - Thể hiện tỷ lệ kiểm soát bóng của mỗi đội
                - Giúp đánh giá khả năng kiểm soát trận đấu

                #### Vùng Hoạt Động
                - Cho thấy phạm vi hoạt động của mỗi đội trên sân
                - Phản ánh chiến thuật và cách triển khai lối chơi

                #### Phân Tích Đội Hình
                - So sánh độ rộng và độ sâu đội hình giữa hai đội
                - Giúp hiểu rõ hơn về chiến thuật và cách xây dựng lối chơi
            """)


def show_page_5():
    st.title("🔧 Trang cài đặt")

# Sidebar for page navigation
pages = {
    "🏠 Giới thiệu": show_introduction,
    "🎥 Khởi chạy mô hình nhận diện": show_player_detection_page,
    "📊 Biểu đồ": show_maps,
    "📈 Các thông số": show_page_4,
    "🔧 Trang cài đặt": show_page_5
}

# Sidebar navigation
page_selection = st.sidebar.radio("Lựa chọn", list(pages.keys()))
page_function = pages[page_selection]
page_function()