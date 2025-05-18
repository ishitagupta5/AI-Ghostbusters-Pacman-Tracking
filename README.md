# Ghostbusters Pacman AI

This project is a modified version of the classic Pacman game where Pacman must track and capture invisible ghosts using noisy sensor data and probabilistic inference. Developed as part of UC Berkeley’s CS188: Introduction to Artificial Intelligence course, this project includes implementations of exact inference, particle filtering, and joint tracking, along with visualization tools and a complete autograding framework.

## Features

- Exact Inference using Bayes Rule
- Particle Filtering for sampling-based belief tracking
- Joint Particle Filtering for multiple ghosts
- Belief distributions displayed graphically and in text
- Ghostbusters mode for inference-based gameplay
- Full autograder and testing infrastructure

## File Overview

| File | Description |
|------|-------------|
| `busters.py` | Game engine for Ghostbusters mode |
| `busters_agents.py` | Pacman agents that use inference modules |
| `busters_ghost_agents.py` | Ghost agents specific to Busters mode |
| `inference.py` | Core inference algorithms (Exact, Particle, JointParticle) |
| `distance_calculator.py` | Precomputes maze distances for inference |
| `ghost_agents.py` | Generic ghost behaviors |
| `keyboard_agents.py` | Keyboard-controlled Pacman agent |
| `layout.py` | Maze configuration parsing |
| `pacman.py` | Core Pacman game logic |
| `game.py` | Game mechanics and state transitions |
| `graphics_display.py` | GUI for Pacman gameplay |
| `text_display.py` | Text-based game display fallback |
| `graphics_utils.py` | Low-level GUI primitives |
| `util.py` | Utility classes like Queue, Counter, PriorityQueue |
| `autograder.py` | Main test runner |
| `submission_autograder.py` | Autograder wrapper for submissions |
| `grading.py` | Scoring and message logging |
| `test_classes.py` | Question and test case definitions |
| `tracking_test_classes.py` | Inference-specific test cases |
| `test_parser.py` | Parses `.test` files for autograder |
| `project_params.py` | Metadata for the project versioning |

## How to Run

### Play Ghostbusters Pacman

Run with keyboard control:
```bash
python busters.py -p BustersKeyboardAgent -l smallGrid
```

Run with exact inference:
```bash
python busters.py -p BustersAgent --inference ExactInference -l smallClassic
```

Run with particle filtering:
```bash
python busters.py -p BustersAgent --inference ParticleFilter -l mediumClassic
```

Visualize belief distributions:
```bash
python busters.py -p BustersAgent --inference ExactInference --display -l mediumClassic
```
### Run Autograder

Run all questions:
```bash
python autograder.py
```

Run a specific question (e.g., q4):
```bash
python autograder.py -q q4
```

## Key Concepts

- **Belief Distributions**: Pacman maintains a probability distribution over ghost locations.
- **Observe**: Pacman updates beliefs based on noisy sensor input.
- **Elapse Time**: Pacman predicts how beliefs evolve over time.
- **Exact vs Particle**: Exact inference calculates probabilities directly; particle filtering approximates using samples.
