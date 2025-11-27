# Contributing to Brain Tumor Classification System

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code:
- Be respectful and inclusive
- Accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs
Before creating bug reports, please check existing issues. When creating a bug report, include:
- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior** vs. actual behavior
- **Screenshots** if applicable
- **Environment details** (OS, Python version, library versions)

### Suggesting Enhancements
Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:
- **Use a clear title** describing the enhancement
- **Provide detailed description** of the proposed functionality
- **Explain why** this enhancement would be useful
- **Include examples** of how it would work

### Code Contributions

We welcome contributions in these areas:

1. **Feature Engineering**
   - New feature extraction methods (GLCM, LBP, HOG)
   - Advanced preprocessing techniques
   - Feature selection algorithms

2. **Model Improvements**
   - Hyperparameter tuning
   - Ensemble methods
   - Alternative algorithms (XGBoost, Neural Networks)

3. **Visualization**
   - New visualization types
   - Interactive plots
   - Model interpretation tools (LIME, SHAP)

4. **Documentation**
   - Code documentation
   - Tutorial notebooks
   - Use case examples

5. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks

## Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Unix/MacOS)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

### 4. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

## Pull Request Process

### Before Submitting
- [ ] Code follows the project's coding standards
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] No unnecessary files are included

### Submission Steps

1. **Update your fork**
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/brain-tumor-classification.git
   git fetch upstream
   git merge upstream/main
   ```

2. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: Add new feature description"
   ```

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
How has this been tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Coding Standards

### Python Style Guide
We follow PEP 8 with some modifications:

```python
# Good
def extract_features(images, verbose=True):
    """
    Extract handcrafted features from images.

    Args:
        images (np.ndarray): Array of images (N, H, W, C)
        verbose (bool): Print progress information

    Returns:
        np.ndarray: Feature array (N, num_features)
    """
    features = []

    for img in images:
        # Extract features here
        feature_vector = process_image(img)
        features.append(feature_vector)

    return np.array(features)
```

### Naming Conventions
- **Variables**: `snake_case` (e.g., `feature_vector`, `train_data`)
- **Functions**: `snake_case` (e.g., `extract_features`, `load_data`)
- **Classes**: `PascalCase` (e.g., `FeatureExtractor`, `ModelTrainer`)
- **Constants**: `UPPER_CASE` (e.g., `IMG_SIZE`, `CLASSES`)

### Code Organization
```python
# 1. Imports (grouped and sorted)
import os
import sys

import numpy as np
import pandas as pd

from sklearn.svm import SVC

# 2. Constants
IMG_SIZE = (224, 224)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 3. Functions
def load_data():
    pass

# 4. Main execution
if __name__ == '__main__':
    main()
```

### Documentation
- All functions must have docstrings
- Use Google-style or NumPy-style docstrings
- Include type hints when possible

```python
def train_svm(X_train: np.ndarray, y_train: np.ndarray,
              C: float = 10.0) -> SVC:
    """
    Train SVM classifier with RBF kernel.

    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
        C: Regularization parameter

    Returns:
        Trained SVM model

    Raises:
        ValueError: If X_train and y_train have mismatched lengths
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have same length")

    model = SVC(kernel='rbf', C=C, random_state=42)
    model.fit(X_train, y_train)

    return model
```

## Testing Guidelines

### Unit Tests
```python
import unittest
import numpy as np

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (224, 224, 3))

    def test_feature_shape(self):
        """Test that features have correct shape"""
        features = extract_features([self.test_image])
        self.assertEqual(features.shape, (1, 20))

    def test_feature_range(self):
        """Test that features are properly normalized"""
        features = extract_features([self.test_image])
        self.assertTrue(np.all(features >= 0))

if __name__ == '__main__':
    unittest.main()
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_features.py

# Run with coverage
python -m pytest --cov=./ tests/
```

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Examples
```bash
feat: Add GLCM texture features extraction
fix: Correct confusion matrix normalization
docs: Update installation instructions
refactor: Simplify feature extraction pipeline
test: Add unit tests for data loading
```

## Questions?

If you have questions:
- Open an issue with the `question` label
- Email: your.email@example.com
- Check existing issues and discussions

## Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Credited in academic papers (if applicable)

---

Thank you for contributing! 
