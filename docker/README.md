# Grand Challenge container

This directory is derived from the official ISLES'26 Docker template and
contains the frozen **ResEnc-L RASS SWA + Primus-M Local-Refinement SWA**
inference path.

The image contains code and dependencies only. Grand Challenge mounts the
separate Model resource at `/opt/ml/model`; model weights must not be copied
into Docker layers.

## Build

```bash
bash do_build.sh
```

The resulting image is `linux/amd64` and tagged
`isles26-strokefusion-final`.

## Local invocation test

Place one native-space T1 file under:

```text
test/input/interf0/images/t1-brain-mri/
```

Update `test/input/interf0/stroke-metadata.json`, then run on a Linux host with
the NVIDIA container runtime:

```bash
bash do_test_run.sh
```

If the two expected checkpoint directories are absent from `model/`, the test
script stages the frozen Model archive from
`../submission_artifacts/model-postprocess.tar.gz` into an ignored temporary
directory. Outputs are written under `test/output/`.

## Export upload resources

```bash
bash do_save.sh
```

This exports the image and independently packages the Model resource through
`../scripts/package_final_model.sh`. Upload the two files to their respective
Grand Challenge resource types.

The runtime expects a T1 MRI plus stroke metadata and writes both a binary
stroke-lesion segmentation and a float lesion-probability map in native space.
