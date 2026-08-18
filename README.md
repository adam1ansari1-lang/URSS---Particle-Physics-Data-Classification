# URSS - Particle-Physics-Data-Classification

Comparing a classical algorithm (RBF SVM) against a quantum algorithm (QVSM) in classifying signals and noise within large particle data sets. The goal is to try and preserve as many signals as possible and get rid of background noise, to then analyse the meaning of these particle interactions and see how we can learn from it. We will be using machine learning methods (decision trees) to build these algorithms. All done via Python.

## Setup

1. Clone the repository: git clone https://github.com/adam1ansari1-lang/URSS---Particle-Physics-Data-Classification.git then cd into it.
2. Create a virtual environment: py -3.13 -m venv venv
3. Activate it: venv\Scripts\Activate.ps1
4. Install dependencies: pip install -r requirements.txt

## Running Scripts

With the virtual environment active, run any script from the project root, for example: python circuit_test.py

## Where Output Goes

plots/ holds generated figures (PNG). results/ holds the raw numbers behind every figure (arrays, CSV). notes/ holds weekly methods write-ups.

## Committing Changes

git add . then git commit -m "describe what changed" then git push

The virtual environment (venv/) is excluded from version control via .gitignore — rebuild it locally using the Setup steps above.