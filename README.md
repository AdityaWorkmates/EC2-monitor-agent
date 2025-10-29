# AWS-Assistant-agent

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![uv](https://img.shields.io/badge/powered%20by-uv-green)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![AWS](https://img.shields.io/badge/cloud-AWS-orange)
![AWS Bedrock](https://img.shields.io/badge/Powered%20by-AWS%20Bedrock-ff9900)
![Strands SDK](https://img.shields.io/badge/SDK-Strands-purple)

---

## 1. Description

The AWS-Monitor-Agent is a versatile intelligent assistant designed to help users monitor and manage their AWS resources. Leveraging AWS CloudWatch metrics, it provides actionable insights into key performance indicators such as CPU utilization, memory usage, and other vital metrics. The agent supports a wide range of AWS services, including but not limited to EC2 instances, RDS databases, Lambda functions, S3 buckets and more. With its ability to list resources, fetch detailed metrics, and visualize data through intuitive plots, the AWS-Monitor-Agent simplifies resource monitoring and management, enhancing operational efficiency across your AWS account.

This tool supports a wide array of AWS services, including but not limited to:
- **EC2 Instances**: Monitor instance performance, CPU utilization, and network activity.
- **RDS Databases**: Gain insights into database metrics like connections, read/write throughput, and latency.
- **Lambda Functions**: Track execution time, error rates, and invocation metrics.
- **S3 Buckets**: Monitor storage usage, request counts, and error rates.
and many more!!

Key Features:
- **Resource Listing**: Easily list and identify AWS resources across regions.
- **Metric Fetching**: Retrieve detailed metrics for a wide range of AWS services.
- **Data Visualization**: Generate intuitive plots for metrics to enhance understanding.
- **Cross-Resource Management**: Manage **All** your AWS services from a unified interface.
- **Chat based Resource Management**: Manage your resources using the chat interface easily.

The AWS Assistant Agent is designed to simplify operational workflows, enhance resource monitoring, and provide a seamless experience for AWS account management.

---

## 2. Installation

To set up and run the AWS Assistant Agent, follow these steps:

### 2.1 Prerequisites
- **Python 3.9+**: Ensure that Python 3.9 or later is installed on your system. You can check your Python version using:
    ```bash
    python --version
    ```
- **`uv` Runtime**: The AWS Assistant Agent is powered by the `uv` runtime for dependency management and execution. If you don't have `uv` installed, follow the official installation guide:
    [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

### 2.2 Steps to Install

1. **Clone the Repository**
    Begin by cloning the repository to your local machine and navigating to the project directory:
    ```bash
    git clone https://github.com/AdityaWorkmates/EC2-monitor-agent.git
    cd EC2-monitor-agent
    ```

2. **Run the Application**
    Use the `uv` runtime to launch the application. `uv` will automatically resolve and install dependencies defined in the `pyproject.toml` file.

    - To run the **Command Line Interface (CLI)**:
        ```bash
        uv run aws_agent_cli.py
        ```
    - To launch the **Gradio Web Interface**:
        ```bash
        gradio aws_agent_gradio.py
        ```
    - To start the **Streamlit Web Interface**:
        ```bash
        streamlit run aws_agent_app.py
        ```

---

## 3. Usage

After successfully running the application, you can interact with the AWS Assistant Agent through the command line or the web interfaces (Gradio or Streamlit). The tool is designed to be user-friendly and supports natural language commands for enhanced usability.

### 3.1 Command Line Examples
The following examples demonstrate typical use cases:

- **List Instances**:
    ```bash
    list instances in us-east-1
    ```
    This command will display all EC2 instances in the specified region (`us-east-1`).

- **Fetch Metrics**:
    ```bash
    show cpu for i-0123456789abcdef in the last 6 hours
    ```
    This command retrieves and displays CPU utilization metrics for the given instance ID (`i-0123456789abcdef`) over the past 6 hours.

- **Generate Plots**:
    ```bash
    plot cpu utilization for instance 1
    ```
    This command generates a visualization of the CPU utilization for the specified instance (assumes you have already listed instances).

---

## 4. Caution on Creating Environments

The AWS Assistant Agent is optimized to work with the `uv` runtime, which simplifies dependency resolution and environment management. When you use `uv run`, it automatically handles the installation of all required dependencies listed in the `pyproject.toml` file.

If you choose to manually create a virtual environment instead of using `uv`, ensure the following:
- Activate your virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```
- Install dependencies manually:
    ```bash
    pip install -r requirements.txt
    ```

However, we strongly recommend using `uv run` for a seamless setup and execution experience.

---

## 5. Agent Architecture Diagram

Below is the architecture diagram of the AWS Assistant Agent, which provides a high-level overview of its components and workflow.

![Agent Architecture Diagram](architecture.png)

### Key Components:
1. **Input Interface**:
    - Command Line Interface (CLI)
    - Web Interfaces (Gradio/Streamlit)

2. **Core Processing**:
    - Natural Language Processing (for understanding user commands)
    - AWS SDK Integration (Boto3/Strands SDK for interacting with AWS services)

3. **Data Visualization**:
    - Interactive Plots (via libraries like Matplotlib/Plotly)
    - Tabular Data Representations

4. **Output**:
    - Console Outputs (for CLI)
    - Dynamic Visualizations (for Gradio/Streamlit)

---

## 6. Supported AWS Services

The AWS Assistant Agent provides extensive support for multiple AWS services, ensuring comprehensive monitoring and management, including but not limited to:
- **Elastic Compute Cloud (EC2)**: Monitor and manage instances.
- **Relational Database Service (RDS)**: Track database performance.
- **Simple Storage Service (S3)**: Manage and monitor bucket usage.
- **Lambda Functions**: Analyze function performance metrics.
- **Elastic Load Balancer (ELB)**: Visualize traffic and latency metrics.

---

## 7. Troubleshooting

### Common Issues:
1. **`uv` not installed**:
    - Ensure you have installed `uv` according to the official documentation: [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

2. **Dependencies not resolving**:
    - Use `uv` to handle dependencies automatically:
        ```bash
        uv run aws_agent_cli.py
        ```

3. **AWS Credentials Missing**:
    - Ensure your AWS credentials are configured. You can set them up using the AWS CLI:
        ```bash
        aws configure
        ```

### Logs:
For debugging and troubleshooting, check the application logs generated in the working directory.

---
