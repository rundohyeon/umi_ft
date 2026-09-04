# Valve State Classifier — rules v4 / run 1

## Purpose

Online, causal recognition of five valve-manipulation states:

1. `approach`
2. `turning`
3. `endpoint_reached`
4. `task_complete`
5. `error`

The auxiliary head predicts `none`, `turn_no_contact`, `post_contact_drop`, or `other`.

## Input contract

- RGB: 224×224, RGB channel order, approximately 59.94/60 Hz.
- Robot state: position 3-D in metres, axis-angle orientation 3-D in radians, gripper width in metres.
- F/T: 12-D physical-unit wrench at approximately 100 Hz, ordered `[L_Fx,L_Fy,L_Fz,L_Tx,L_Ty,L_Tz,R_Fx,R_Fy,R_Fz,R_Tx,R_Ty,R_Tz]`, with force in N and torque in Nm.
- Call `reset()` at every episode boundary.

At each decision, the classifier samples 16 RGB/robot observations with stride 4, covering approximately 1.001 seconds. Each sampled observation carries the preceding 50 F/T samples, approximately 0.49 seconds. No future input is used.

## Architecture

- Frozen ImageNet ResNet-18: 512 → 128-D.
- Three-layer causal force Conv1d encoder: 12-D wrench plus mask → 128-D.
- Robot-state MLP: 10 → 64 → 64-D.
- Concatenation and projection: 320 → 256-D.
- Three residual causal TCN blocks, kernel 3, two convolutions per block, dilations 1/2/4.
- 5-way phase head and 4-way error-reason head.

## Training snapshot

- Checkpoint: epoch 23 selected by validation `macro-F1 − 0.05 × error FPR`.
- Episode-disjoint split: train 150, validation 32, test 32.
- Held-out test accuracy: 0.9603.
- Held-out test macro-F1: 0.9495.
- Held-out error precision/recall: 0.9141 / 0.9141.

## Important limitation

The supervision is automatic v4 rule-generated pseudo-labeling, not an independently human-annotated ground truth. The reported test metrics therefore measure agreement with v4 labels on held-out episodes. Validate the model on the target computer's camera, robot-state convention, F/T ordering, calibration, and operational distribution before using it for safety-critical control.
