from diagrams import Diagram, Cluster, Edge
# AWS Components
from diagrams.aws.compute import EC2, ECR, ECS
from diagrams.aws.storage import S3
from diagrams.aws.network import ELB
from diagrams.aws.management import Cloudwatch
# On-Premise/Tooling Components
from diagrams.onprem.container import Docker
from diagrams.onprem.database import Postgresql
 # Used for ELK Stack
             # Used for MLOps Pipeline/CI/CD at bottom
from diagrams.onprem.vcs import Git                 # Used for DVC
# Programming Components
from diagrams.programming.framework import Angular, Flask
from diagrams.programming.language import Python
from diagrams.ml.framework import Mlflow
from diagrams.ml.model import Model                 # Used for the brain icon

# Graph attributes for a cleaner layout
graph_attr = {
    "splines": "spline",
    "nodesep": "0.6",
    "ranksep": "1.2",
    "fontname": "Sans-Serif",
    "bgcolor": "white"
}

with Diagram("Production MLOps Architecture (Strict Match)", show=True, direction="LR", graph_attr=graph_attr):

    # --- Front End Cluster ---
    with Cluster("Front end", graph_attr={"pencolor": "#d9534f", "style": "dashed", "fontcolor": "#d9534f"}):
        angular = Angular("Angular")
        docker_fe = Docker("docker")

    # --- Back End Cluster ---
    with Cluster("Back end"):
        # Data & Experimentation Group
        data = Database("Data")
        jupyter = Python("jupyter")
        dvc = Git("DVC")
        s3 = S3("Amazon S3")
        mlflow = Mlflow("MLflow")
        ec2_data = EC2("Amazon EC2")

        # Model Development Group
        model = Model("model")
        pytest_tool = Python("pytest")
        flask_api = Flask("Flask\nRestful API")
        docker_be = Docker("docker")

        # Registry & Deployment Group
        ecr = ECR("Amazon ECR")
        ecs = ECS("AWS ECS")

        webapp = Html5("Web app")

       
      
        dash = Dashboard("Dashboard")

        # MLOps Pipeline (CI/CD) at the bottom
        cicd = CI_CD("MLOps Pipeline\nCI/CD")

        # --- Connections ---
        # Data Flow
        data >> jupyter >> dvc >> s3
        jupyter >> [mlflow, model]
        mlflow >> ec2_data
        
        # Model Training & API Flow
        model >> mlflow
        model >> pytest_tool >> flask_api >> docker_be >> ecr

        # Deployment Flow (Ports are explicit in the diagram)
        ecr >> Edge(label=":4200") >> ecs
        ecr >> Edge(label=":5000") >> ecs
        ecs >> lb >> webapp

        # Monitoring Flow
        ecs >> cw >> dash
        ecs >> elk >> dash

        # --- Cross-Cluster and Pipeline Connections ---
        angular >> docker_fe >> ecr
        # CI/CD orchestrates all major components (represented by the upward arrows)
        cicd >> [data, jupyter, model, flask_api, cw, elk, dash, webapp, angular]