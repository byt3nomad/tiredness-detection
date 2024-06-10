#[macro_use]
extern crate cpp;

mod wrapper;

use dlib_face_recognition::ImageMatrix;
use opencv::prelude::Mat;
use opencv::prelude::MatTraitConst;

/// Copy a matrix from an opencv mat, supporting both color and grayscale images
pub fn matrix_to_opencv_mat(mat: &Mat) -> ImageMatrix {
    let mat = mat.as_raw_Mat();

    unsafe {
        cpp!([mat as "const cv::Mat*"] -> ImageMatrix as "dlib::matrix<dlib::rgb_pixel>" {
            // Check the number of channels in the input image
            if (mat->channels() == 1) {
                // Handle grayscale image
                dlib::cv_image<unsigned char> gray_image(*mat);
                dlib::matrix<dlib::rgb_pixel> out;

                // Convert grayscale to RGB by assigning the gray value to each color channel
                dlib::assign_image(out, gray_image);
                return out;
            } else {
                // Handle color image
                dlib::cv_image<dlib::bgr_pixel> color_image(*mat);
                dlib::matrix<dlib::rgb_pixel> out;

                // Convert BGR to RGB
                dlib::assign_image(out, color_image);
                return out;
            }
        })
    }
}
