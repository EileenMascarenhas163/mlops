from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2
from diagrams.aws.storage import S3
from diagrams.onprem.client import User
from diagrams.onprem.vcs import Github
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.container import Docker
from diagrams.onprem.monitoring import Prometheus, Grafana
from diagrams.onprem.database import Postgresql

from diagrams.programming.framework import FastAPI
from diagrams.programming.language import Python

with Diagram(
    "Production MLOps Architecture: Loan Approval System (CI/CD/CT/CM)",
    show=True,
    direction="TB",
    filename="mlops_professional_architecture"
):
    # Users
    end_user = User("Client Portal User")
    ml_engineer = User("MLOps Engineer")

    # CI/CD
    with Cluster("Development & Continuous Integration"):
        github = Github("Source Code Repository")
        cicd = GithubActions("CI/CD Automation Pipeline")
        ml_engineer >> github >> cicd

    cicd_target = EC2("AWS ECS/EKS Compute Cluster")
    cicd >> Edge(label="Container Deployment") >> cicd_target

    with Cluster("Production MLOps Services"):
        # Training Pipeline (Continuous Training - CT)
        with Cluster("Continuous Training (CT)"):
            dataset = S3("Feature Store / Data Lake (S3)")
            etl = Python("Data Prep & Feature Engineering")
            trainer = Docker("Sklearn/XGBoost Model Trainer")
            dataset >> etl >> trainer

        # Model Management
        mlflow = Postgresql("MLflow Tracking & Registry\n(Model Governance)")
        trainer >> Edge(label="Log Experiment & Model Artifact") >> mlflow

        # Serving Layer (Continuous Deployment - CD)
        with Cluster("Low-Latency Model Serving"):
            api = FastAPI("Inference API Gateway (/predict)")
           
            ui >> Edge(label="HTTP Request") >> api
            api >> Edge(label="Load Model via @prod Alias") >> mlflow

        # Monitoring (Continuous Monitoring - CM)
        with Cluster("Model Observability & Monitoring"):
            prometheus = Prometheus("Runtime Metrics Scraper")
            grafana = Grafana("Performance & Drift Dashboards")
            api >> Edge(label="Expose & Scrape Metrics") >> prometheus
            prometheus >> grafana

        # Link core components to the compute cluster
        cicd_target >> [etl, trainer, mlflow, api, ui, prometheus, grafana]

    # External Access
    end_user >> ui