# UMI context dataset transfer bundle

이 ZIP은 context observer / context-conditioned UMI 학습용 세 항목만 포함합니다.

## 내용

```text
canonical_context_supervision_v2_4state_284.npz
dataset.zarr.zip
dataset_force_sidecar.zarr/
```

## 연결 관계

```text
dataset.zarr.zip                    RGB + robot TCP 10D trajectory (EP 1--284)
dataset_force_sidecar.zarr/         timestamp-aligned left/right 6-axis F/T sidecar
canonical_context_supervision...npz frame-wise A/T/R/E context labels (EP 1--284)
```

context label class order:

```text
0 approach
1 turning
2 recovery
3 error
```

## SHA-256

```text
e812b062d20327ab4cee67f1bca6c4f5feaf266960c9165d4de074635f7da0c4  canonical_context_supervision_v2_4state_284.npz
1687080df176590b3b5e4112c8c24b6a1d25e74fdcf2c7557686e3da9970617a  dataset.zarr.zip
10c6cc41d1ecc9cd0b176a57a1ba97818516335830126d02a845e62bf1114ee5  dataset_force_sidecar.zarr (aggregate of sorted file hashes)
```

`dataset.zarr.zip` itself is already compressed, so this outer transfer ZIP is stored without recompression. Extract it first, then pass the extracted dataset ZIP and extracted sidecar directory as separate paths to the training configuration.
