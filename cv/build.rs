fn main() {
    println!("cargo:rustc-link-lib=dlib");
    println!("cargo:rustc-link-lib=lapack");
    println!("cargo:rustc-link-lib=openblas");

    cpp_build::Config::new()
        .flag_if_supported("-std=c++11") // Ensure C++11 is used for building the C++ code
        .include("/usr/include/opencv4")
        .build("src/lib.rs");
}
