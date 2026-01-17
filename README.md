# Mini-ML-Projects
Here are some fun and creative projects to explore state of the art ML architectures and applications.



# 1. Beret: The Drawing Robot
Beret is a software that explores two common architectures: the transformer and diffusion model. Through this creative project, 1) learn how LLMs like GPT comprehend language through self-attention, 2) learn how image generation models carve out subjects from pure noise using a diffusion process, and 3) walk through inverse kinematics in Beret's personalized G-code generation feature.

## Features
- B1: A transformer class built layer by layer using PyTorch.
- B2: A sample csv dataset to train Beret's transformer. Feel feel to add more data.
- B3: A diffusion model class built layer by layer using PyTorch.
- B4: A sample image dataset for label testing.
- B5: A g-code generator to convert images into joint angle sequences. Build your own two-degree-of-freedom arm to test it out!
- Beret: The full pipeline with UI.
- Other: Expressions for UI, JPGS for B4, and a testing file.

## Usage
- Go through each file in numerical order to gain insight on each of Beret's components.
- Train both models on the provided data or your own datasets.
- Run the test file and make some drawings!
- Note: make sure to create empty folders for the paths at the top of Beret's main program.



# 2. Blackjack
Play by yourself, with friends, or against AI. This program includes a terminal blackjack game with a built-in betting mechanism & a trainable blackjack AI. Players (human or AI) can hit, stand, double, or split while the AI learns optimal strategies using a PyTorch neural network.

## Features
- Full Blackjack gameplay with betting and multiple players.
- AI player that learns from outcomes and uses card counting for strategy.
- Tracks game statistics and plots player balances over rounds.
- Designed as a hands-on playground for ML experimentation.

## Requirements
- Python 3.8+
- PyTorch
- Matplotlib

## Usage
- Enter number of players and names (prefix AI names with "AI", e.g., AI_1).
- Place bets and choose actions (h for hit, s for stand, double, and split).
- Watch the AI learn and adapt over time.
