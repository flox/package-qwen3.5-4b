# mkHfModel.nix — shared builder for HuggingFace model packages (vLLM/Triton)
#
# Triton-only layout (default, slug = null):
#   $out/share/models/<tritonModelName>/
#     config.pbtxt          — standard vLLM Triton backend config
#     model-defaults.json   — vLLM engine defaults (no "model" key; resolved at runtime)
#     weights/              — cp -rL of srcPath contents
#
# Dual layout (slug + snapshotId provided):
#   $out/share/models/hub/models--<slug>/
#     refs/main             — contains snapshotId
#     snapshots/<id>/       — actual model files (cp -rL, once)
#   $out/share/models/<tritonModelName>/
#     config.pbtxt
#     model-defaults.json
#     weights → ../hub/models--<slug>/snapshots/<id>   (relative symlink)
#
# Uses sandbox = "off" so local paths are accessible.
#
# Usage (from per-model .nix files):
#   { pkgs, mkHfModel ? pkgs.callPackage ./mkHfModel.nix {} }:
#   mkHfModel { pname = "..."; baseVersion = "..."; buildMeta = ...; srcPath = /path/to/snapshot;
#               tritonModelName = "..."; vllmDefaults = { ... }; }
#   # For dual layout, also pass slug and snapshotId:
#   mkHfModel { ...; slug = "Qwen--Qwen3.5-4B"; snapshotId = "851bf6e8..."; }
{ stdenv }:
{ pname, baseVersion, buildMeta, srcPath, tritonModelName, vllmDefaults ? {},
  slug ? null, snapshotId ? null }:

let
  version = "${baseVersion}+${buildMeta.git_rev_short}";
  defaultsJson = builtins.toJSON vllmDefaults;
  dualLayout = slug != null && snapshotId != null;
in
stdenv.mkDerivation {
  inherit pname version;
  src = srcPath;
  dontBuild = true;
  installPhase = ''
    _model="$out/share/models/${tritonModelName}"
    mkdir -p "$_model"
  '' + (if dualLayout then ''
    # --- Dual layout: HF cache + Triton with symlinked weights ---
    _snap="$out/share/models/hub/models--${slug}/snapshots/${snapshotId}"
    mkdir -p "$_snap"
    cp -rL $src/snapshots/${snapshotId}/* "$_snap/"
    rm -f "$_snap/.gitattributes" "$_snap/README.md" "$_snap/LICENSE"

    mkdir -p "$out/share/models/hub/models--${slug}/refs"
    echo -n "${snapshotId}" > "$out/share/models/hub/models--${slug}/refs/main"

    ln -s "../hub/models--${slug}/snapshots/${snapshotId}" "$_model/weights"
  '' else ''
    # --- Triton-only layout: weights copied directly ---
    mkdir -p "$_model/weights"
    cp -rL $src/* "$_model/weights/"
  '') + ''

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
