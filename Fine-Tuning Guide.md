graph TB
    subgraph "Data Collection & Preparation"
        A[Gaming Info Collection] --> C[Dataset Creation]
        B[GPT Model] --> C
        C --> D[Semi-Supervised Dataset]
        C --> E[DPO Dataset]
    end
    
    subgraph "Foundation Model"
        F[Qwen 3 32B Parameters<br/>Base Model]
    end
    
    subgraph "Fine-Tuning Pipeline"
        G[Stage 1: SFT Fine-Tuning<br/>qLoRA Adapter 1<br/>(~1% parameters)]
        H[Stage 2: DPO Fine-Tuning<br/>qLoRA Adapter 2<br/>(~1% parameters)]
        
        D --> G
        E --> H
        F --> G
        G --> H
    end
    
    subgraph "Specialized Fine-Tuning Targets"
        I[Community Details<br/>Understanding]
        J[PG Rating<br/>Classification]
        K[Topic Level<br/>Accuracy]
        L[Severity<br/>Accuracy]
        
        H --> I
        H --> J
        H --> K
        H --> L
    end
    
    subgraph "Enhanced Model Output"
        M[Gaming Info Enhancement<br/>Foundation Layer]
        
        I --> M
        J --> M
        K --> M
        L --> M
    end
    
    subgraph "Technical Implementation"
        N[qLoRA Adapters<br/>• Low-rank adaptation<br/>• 1% parameter training<br/>• Memory efficient<br/>• Faster training]
        
        G -.-> N
        H -.-> N
    end
    
    style F fill:#e1f5fe
    style M fill:#c8e6c9
    style N fill:#fff3e0
    style C fill:#f3e5f5
