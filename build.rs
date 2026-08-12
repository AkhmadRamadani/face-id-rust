fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target = std::env::var("TARGET").unwrap_or_default();

    // Ensure the linker can locate libLiteRt dynamic libraries at build time and runtime via RPATH.
    if let Some(cache_dir) = dirs::cache_dir() {
        let litert_dir = cache_dir
            .join("litert-sys")
            .join("v0.10.2")
            .join(&target);
        let dir_str = litert_dir.display();
        println!("cargo:rustc-link-search=native={dir_str}");
        if target.contains("apple") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir_str}");
        } else if target.contains("linux") {
            println!("cargo:rustc-link-arg=-Wl,-rpath={dir_str}");
        }
    }
}
