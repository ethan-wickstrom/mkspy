# mkspy

Meta-optimization framework for generating DSPy programs. This project provides:

- `DSPyProgramGenerator`: a meta-module that produces complete DSPy programs from task descriptions.
- `ProgramGenerationMetric`: rich metric with feedback for GEPA.
- `GEPAEvolver`: optimizer using GEPA as the sole evolution strategy.
- Smart mutation strategies responding to feedback.
- A small task library for experiments.
