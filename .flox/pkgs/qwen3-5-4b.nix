# Qwen3.5-4B — multimodal vision+text model, BF16, dual layout
#
# 4B parameters, 8.8 GB on disk.
# Architecture: Qwen3_5ForConditionalGeneration (vision + text)
{ pkgs, stdenv }:

let
  buildMeta = builtins.fromJSON (builtins.readFile ../../build-meta/qwen3.5-4b.json);

  # --- fetch split tarball parts from GitHub Releases ---
  fetchedParts = map (p: pkgs.fetchurl { inherit (p) url hash; }) [
    { url = "https://github.com/flox/package-qwen3.5-4b/releases/download/qwen3.5-4b-v1.0/qwen3.5-4b.tar.part-aa"; hash = "sha256-SgZV1cv6leuwY8c8JDHvmqSw6H2pRIW2o/lipE4m5C8="; }
    { url = "https://github.com/flox/package-qwen3.5-4b/releases/download/qwen3.5-4b-v1.0/qwen3.5-4b.tar.part-ab"; hash = "sha256-a0EAAes/aU1HEdxMpo3xxVcEh8v23M4ItlNpbcwmfWI="; }
    { url = "https://github.com/flox/package-qwen3.5-4b/releases/download/qwen3.5-4b-v1.0/qwen3.5-4b.tar.part-ac"; hash = "sha256-qNSvLi2aEs3aiz4q+B3X3pjXBw08FjmhCuTepzQ62jc="; }
    { url = "https://github.com/flox/package-qwen3.5-4b/releases/download/qwen3.5-4b-v1.0/qwen3.5-4b.tar.part-ad"; hash = "sha256-7okyqR16z6tNLyMuXLtTlsNc5f4nfZQXyPQaFZMMJ34="; }
    { url = "https://github.com/flox/package-qwen3.5-4b/releases/download/qwen3.5-4b-v1.0/qwen3.5-4b.tar.part-ae"; hash = "sha256-85uAX/Amfhm275cFjIZvQ/XxI4/nXyH0f9C2m/AG9ro="; }
  ];
  partPaths = pkgs.lib.concatMapStringsSep " " toString fetchedParts;

  modelSrc = pkgs.runCommand "qwen3.5-4b-src" {} ''
    mkdir -p $out
    cat ${partPaths} | tar xf - -C $out
  '';

  # --- package config ---
  pname = "qwen3.5-4b";
  baseVersion = "1.0.0";
  version = "${baseVersion}+${buildMeta.git_rev_short}";
  tritonModelName = "qwen3_5_4b";
  slug = "Qwen--Qwen3.5-4B";
  snapshotId = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";

  vllmDefaults = {
    gpu_memory_utilization = 0.85;
    max_model_len = 4096;
    dtype = "bfloat16";
    enable_log_requests = false;
  };
  defaultsJson = builtins.toJSON vllmDefaults;

in
stdenv.mkDerivation {
  inherit pname version;
  src = "${modelSrc}/models--Qwen--Qwen3.5-4B";
  dontBuild = true;
  installPhase = ''
    _model="$out/share/models/${tritonModelName}"
    mkdir -p "$_model"

    # --- Dual layout: HF cache + Triton with symlinked weights ---
    _snap="$out/share/models/hub/models--${slug}/snapshots/${snapshotId}"
    mkdir -p "$_snap"
    cp -rL $src/snapshots/${snapshotId}/* "$_snap/"
    rm -f "$_snap/.gitattributes" "$_snap/README.md" "$_snap/LICENSE"

    mkdir -p "$out/share/models/hub/models--${slug}/refs"
    echo -n "${snapshotId}" > "$out/share/models/hub/models--${slug}/refs/main"

    ln -s "../hub/models--${slug}/snapshots/${snapshotId}" "$_model/weights"

    cat > "$_model/config.pbtxt" << 'PBTXT'
    backend: "vllm"

    instance_group [
      {
        count: 1
        kind: KIND_MODEL
      }
    ]

    model_transaction_policy {
      decoupled: True
    }

    input [
      {
        name: "text_input"
        data_type: TYPE_STRING
        dims: [ 1 ]
      },
      {
        name: "stream"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        optional: true
      },
      {
        name: "sampling_parameters"
        data_type: TYPE_STRING
        dims: [ 1 ]
        optional: true
      },
      {
        name: "exclude_input_in_output"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        optional: true
      }
    ]

    output [
      {
        name: "text_output"
        data_type: TYPE_STRING
        dims: [ -1 ]
      }
    ]
    PBTXT

    cat > "$_model/model-defaults.json" << 'DEFAULTS'
    ${defaultsJson}
    DEFAULTS

    mkdir -p "$out/share/${pname}"
    echo -n "${version}" > "$out/share/${pname}/flox-build-version-${toString buildMeta.build_version}"
  '';
}
