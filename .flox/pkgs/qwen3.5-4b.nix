# Qwen3.5-4B — multimodal vision+text model, BF16, dual layout
#
# 4B parameters, 8.8 GB on disk.
# Architecture: Qwen3_5ForConditionalGeneration (vision + text)
{ pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {},
  fetchModelRelease ? pkgs.callPackage ./fetchModelRelease.nix {} }:

let
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/qwen3.5-4b.json);
  modelSrc = fetchModelRelease {
    name = "qwen3.5-4b-src";
    parts = [
      { url = "https://github.com/flox/package-qwen3.5-4b/releases/download/qwen3.5-4b-v1.0/qwen3.5-4b.tar.part-aa"; hash = "sha256-SgZV1cv6leuwY8c8JDHvmqSw6H2pRIW2o/lipE4m5C8="; }
      { url = "https://github.com/flox/package-qwen3.5-4b/releases/download/qwen3.5-4b-v1.0/qwen3.5-4b.tar.part-ab"; hash = "sha256-a0EAAes/aU1HEdxMpo3xxVcEh8v23M4ItlNpbcwmfWI="; }
      { url = "https://github.com/flox/package-qwen3.5-4b/releases/download/qwen3.5-4b-v1.0/qwen3.5-4b.tar.part-ac"; hash = "sha256-qNSvLi2aEs3aiz4q+B3X3pjXBw08FjmhCuTepzQ62jc="; }
      { url = "https://github.com/flox/package-qwen3.5-4b/releases/download/qwen3.5-4b-v1.0/qwen3.5-4b.tar.part-ad"; hash = "sha256-7okyqR16z6tNLyMuXLtTlsNc5f4nfZQXyPQaFZMMJ34="; }
      { url = "https://github.com/flox/package-qwen3.5-4b/releases/download/qwen3.5-4b-v1.0/qwen3.5-4b.tar.part-ae"; hash = "sha256-85uAX/Amfhm275cFjIZvQ/XxI4/nXyH0f9C2m/AG9ro="; }
    ];
  };
in
mkHfModel {
  pname = "qwen3.5-4b";
  baseVersion = "1.0.0";
  inherit buildMeta;
  srcPath = "${modelSrc}/models--Qwen--Qwen3.5-4B";
  tritonModelName = "qwen3_5_4b";

  # Dual layout: vLLM + SGLang + Triton
  slug = "Qwen--Qwen3.5-4B";
  snapshotId = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";

  vllmDefaults = {
    gpu_memory_utilization = 0.85;
    max_model_len = 4096;
    dtype = "bfloat16";
    enable_log_requests = false;
  };
}
