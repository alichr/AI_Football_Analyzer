# 🏗️ System Architecture Diagrams

This document contains detailed flow diagrams for each major component of the AI-Based Football Analyzer system.

## Data Pipeline Flow
```mermaid
graph LR
    direction LR
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#ccf,stroke:#333,stroke-width:2px
    A(Raw Video Files) --> B[Frame Extraction]
    B --> C[Resize/Normalize]
    C --> D[Metadata Extraction]
    D --> E(Processed Data)
    subgraph Input
        direction TB
        A
    end
    subgraph Output
        direction TB
        E
        E --- F(Timestamps)
        E --- G(Video Metadata)
    end
```

## Detection & Tracking Flow
```mermaid
flowchart LR
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px
    A(Processed Frames) --> B{"Player/Ball Detection\n(YOLOv8)"}
    B --> C[Bounding Boxes]
    A --> D{"Tracking\n(DeepSORT)"}
    D --> E[Player IDs, Trajectories]
    C & E --> F(Tracklets & Annotated Video)
    subgraph Input
        direction TB
        A
    end
    subgraph Output
        direction TB
        F
    end
```

## Strategy Analysis Flow
```mermaid
graph LR
    direction LR
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style E fill:#ccf,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:2px
    A(Tracklets) --> B1[Heatmap Generation] -- Output --> C(Heatmap Images)
    A --> B2[Passing Network] -- Output --> E(Passing Graphs)
    A --> B3[Formation Detection] -- Output --> G(Formation Labels)
    subgraph Input
        direction TB
        A
    end
    subgraph Output
        direction TB
        C
        E
        G
    end
```

## API & Web App Flow
```mermaid
graph LR
    direction LR
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#ccf,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:2px
    subgraph Input
        direction LR
        A(Analysis Results)
        B(User Requests)
    end
    A & B --> C{FastAPI: REST Endpoints} -- Output --> E(Interactive Visualizations)
    A & B --> D{Streamlit/React: Dashboard} -- Output --> E
    E -- Output --> F(Downloadable Reports)
```

## MLOps & Deployment Flow
```mermaid
graph LR
    direction LR
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style EE fill:#ccf,stroke:#333,stroke-width:2px
    subgraph Input
        direction LR
        A(Model Artifacts)
        B(Codebase)
    end
    A & B --> C[CI/CD] --> EE(Deployed API)
    A & B --> D[Docker/Kubernetes] --> EE
    A & B --> F[MLflow] --> G[Model Versioning]
    EE --> H(Monitoring Logs)
``` 