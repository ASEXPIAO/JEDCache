# JEDCache: core controller

Actual core implementation for **JEDCache: Joint-Error-Driven Caching for Faster Four-Branch Video Editing**.

JEDCache predicts error in the signed, combined four-branch editing direction and shares a refresh decision across four separate deep-residual caches. `jedcache/joint_cache.py` is the unchanged controller used in the authors' experiments. Historical `Joint` / `joint` identifiers denote JEDCache.

## First-release scope

This release contains the controller/cache implementation, its internal control policies, and CPU API tests with synthetic tensors and a synthetic fit. It is executable core code, not a placeholder repository.

Experimental calibration observations, frozen fitted parameters, result datasets, human-rating records, source videos, model weights and the complete Wan-Edit generation runner are not included in this code-only release. The supplied tests do not reproduce paper results or run video generation. More integration documentation is planned. No journal acceptance is claimed.

## Install and test

Use Python 3.10 or newer and install PyTorch for your platform. From this directory:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

See [integration notes](docs/INTEGRATION.md). Four controller tests passed locally on CPU using an existing Python 3.10 / PyTorch 1.12.1+cu113 environment; this establishes core compatibility only. New installations are directed to PyTorch 2.x. The paper's generation environment and measurements are separate from these synthetic API checks.

## License and attribution

Original code is released under the MIT license. PyTorch and separately obtained backbones, datasets and checkpoints retain their own licenses. No third-party backbone or model checkpoint is redistributed here. The controller originated in the authors' FiVE-Bench Wan-Edit experimental integration; consult the accompanying manuscript for citations to FiVE-Bench, Wan2.1 and FlowEdit.

Source-controller SHA-256: `b74a905f2bc0409fe707b9029e188bc503df81b4ac1a0b13b7f212596cc0c86d`.
