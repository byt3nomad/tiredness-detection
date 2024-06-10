use core::BlinkDetector;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let detector = BlinkDetector::new().expect("Failed to initialize the blink detector");
    detector.tiredneess_level("./many.mp4");
    Ok(())
}
