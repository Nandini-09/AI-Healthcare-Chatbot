# Project Setup

This README provides instructions for setting up and running the project.

## Prerequisites

- Python 3.x installed on your machine.
- Ensure `pip` is available for managing Python packages.

## Setup Instructions

1. **Create a Virtual Environment:**
   ```bash
   pip -m venv venv
   ```

2. **Install Project Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**
   - Add your Gemini API key to the `.env` file in the following format:
     ```env
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

## Running the Project

To start the project, run the following command:

```bash
streamlit run app.py
```

This will launch the application in your default web browser.

## Notes

- Ensure that the `.env` file is properly configured with your API key before running the project.
- Activate the virtual environment before running the application to ensure the correct dependencies are used.
