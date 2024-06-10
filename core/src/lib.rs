use dlib_face_recognition::*;
use eye_analysis::{get_blinks_from_vec, get_frame_ear, proportion_below_threshold};
use opencv::{
    core::{rotate, RotateFlags, Size},
    highgui, imgproc,
    prelude::*,
    videoio,
};

mod eye_analysis;

const MAX_BLINKS_PER_4_SECONDS: f64 = 4.0;
const MAX_PROPORTION_BELOW_THRESHOLD: f64 = 0.5;

#[derive(Clone)]
pub struct BlinkDetector {
    face_detector: FaceDetectorCnn,
    landmark_predictor: LandmarkPredictor,
}

impl BlinkDetector {
    pub fn new() -> Option<Self> {
        let face_detector = FaceDetectorCnn::default().ok()?;
        let landmark_predictor = LandmarkPredictor::default().ok()?;
        highgui::named_window("video", highgui::WINDOW_AUTOSIZE).ok()?;

        Some(BlinkDetector {
            face_detector,
            landmark_predictor,
        })
    }

    pub fn tiredness_level(&self, file_path: &str) -> Option<f64> {
        let ears = Self::get_video_ear(&self.face_detector, &self.landmark_predictor, file_path)?;

        let total_blinks = get_blinks_from_vec(&ears);
        let proportion_below_threshold = proportion_below_threshold(&ears);

        println!("Total Blinks: {}", total_blinks);
        println!(
            "Proportion Below Threshold: {:.2}%",
            proportion_below_threshold * 100.0
        );
        let blink_score = (total_blinks as f64 / MAX_BLINKS_PER_4_SECONDS).min(1.0);
        let proportion_score =
            (proportion_below_threshold / MAX_PROPORTION_BELOW_THRESHOLD).min(1.0);

        println!("proportion score {}", proportion_score);
        println!("proportion below {}", proportion_below_threshold);
        println!("blink score {}", blink_score);

        let tiredness_level = (blink_score + proportion_score) / 2.0 * 100.0;
        println!("Tiredness level: {}", tiredness_level);
        Some(tiredness_level)
    }

    fn get_video_ear(
        face_detector: &FaceDetectorCnn,
        landmark_predictor: &LandmarkPredictor,
        file_path: &str,
    ) -> Option<Vec<f64>> {
        let mut ears: Vec<f64> = Vec::new();
        let mut cam = videoio::VideoCapture::from_file(file_path, videoio::CAP_FFMPEG).ok()?;
        cam.set(videoio::CAP_PROP_ORIENTATION_AUTO, 0.0).ok()?;
        let mut frame = Mat::default();
        let rotation_flag: RotateFlags = RotateFlags::ROTATE_90_CLOCKWISE;

        while videoio::VideoCapture::read(&mut cam, &mut frame).ok()? {
            if frame.size().ok()?.width == 0 {
                break;
            }

            let mut rotated_frame = Mat::default();
            rotate(&frame, &mut rotated_frame, rotation_flag as i32).ok()?;
            frame = rotated_frame;

            let avg_ear = get_frame_ear(face_detector, landmark_predictor, &mut frame)?;
            println!("EAR: {avg_ear}");
            ears.push(avg_ear);
        }

        Some(ears)
    }
}
