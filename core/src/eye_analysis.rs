use dlib_face_recognition::*;
use dlib_face_recognition_cv::matrix_to_opencv_mat;
use opencv::{
    core::{Point, Scalar},
    highgui, imgproc,
    prelude::*,
};

const LEFT_EYE_IDS: &[usize] = &[36, 37, 38, 39, 40, 41];
const RIGHT_EYE_IDS: &[usize] = &[42, 43, 44, 45, 46, 47];

fn distance(p1: Point, p2: Point) -> f64 {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    ((dx * dx + dy * dy) as f64).sqrt()
}

fn calculate_ear(eye_points: &[Point]) -> f64 {
    let vertical1 = distance(eye_points[1], eye_points[5]);
    let vertical2 = distance(eye_points[2], eye_points[4]);
    let horizontal = distance(eye_points[0], eye_points[3]);
    (vertical1 + vertical2) / (2.0 * horizontal)
}

fn extract_face_points(landmarks: &FaceLandmarks, eye_ids: &[usize]) -> Vec<Point> {
    eye_ids
        .iter()
        .map(|&id| {
            let point = landmarks.get(id).expect("Invalid landmark index");
            Point::new(point.x() as i32, point.y() as i32)
        })
        .collect()
}

pub fn proportion_below_threshold(ears: &[f64]) -> f64 {
    let below_threshold_count = ears.iter().filter(|&&ear| ear < 0.27).count();
    below_threshold_count as f64 / ears.len() as f64
}

pub fn get_blinks_from_vec(ears: &[f64]) -> i32 {
    let mut blink_count = 0;
    let mut frames_below_threshold = 0;

    for &ear in ears {
        if ear < 0.27 {
            frames_below_threshold += 1;
        } else {
            if frames_below_threshold >= 3 {
                blink_count += 1;
            }
            frames_below_threshold = 0;
        }
    }

    blink_count
}

pub fn get_frame_ear(
    face_detector: &FaceDetectorCnn,
    landmark_predictor: &LandmarkPredictor,
    frame: &mut Mat,
) -> Option<f64> {
    let matrix = matrix_to_opencv_mat(frame);

    let faces = face_detector.face_locations(&matrix);
    if faces.len() != 1 {
        return None; //We need only 1 face
    }
    let first_face = &faces[0];

    let landmarks = landmark_predictor.face_landmarks(&matrix, first_face);
    let left_eye_points = extract_face_points(&landmarks, LEFT_EYE_IDS);
    let right_eye_points = extract_face_points(&landmarks, RIGHT_EYE_IDS);

    let ear_left = calculate_ear(&left_eye_points);
    let ear_right = calculate_ear(&right_eye_points);
    let avg_ear = (ear_left + ear_right) / 2.0;

    for eye_point in left_eye_points.iter().chain(right_eye_points.iter()) {
        imgproc::circle(
            frame,
            *eye_point,
            1,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )
        .ok()?;

        highgui::imshow("video", frame).ok()?;
        if highgui::wait_key(10).ok()? > 0 {
            break;
        }
    }

    Some(avg_ear)
}
