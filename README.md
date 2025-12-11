# Cascade Context Protocol (CCP): A Neuro-Symbolic Architecture

The **Cascade Context Protocol (CCP)** is a robust architectural framework designed to align Large Language Models (LLMs) with biological memory systems and decision-making processes. By mimicking the functional specialized regions of the mammalian brain, CCP solves the "Context Drift" and "Tool Selection" problems inherent in naive RAG implementations.

## Theoretical Basis

### Entorhinal Grid Cell Alignment (Vector Normalization)
In the biological brain, the Entorhinal Cortex provides a spatial coordinate system (grid cells) that allows the hippocampus to place memories in a physical or conceptual space. CCP implements this via the **VectorNormalizer**.

Raw embeddings from LLMs are often anisotropic and task-agnostic. The VectorNormalizer acts as a learned transformation layer that "rotates" and aligns these user queries into the correct task-specific semantic manifold. This ensures that a query about "Python code" lands in the precise region of latent space where "Programming Logic" is stored, effective mimicking the orienting function of grid cells.

### Basal Ganglia Action Selection (Softmax Routing)
The brain does not "think" about every action. The Basal Ganglia acts as a gating mechanism, selecting actions based on learned reinforcement signals (dopamine). CCP implements this via the **SoftmaxRouter**.

Instead of performing expensive semantic search for every tool invocation, the SoftmaxRouter serves as a "Learned Intuition" layer. It takes the current context vector and outputs a probability distribution over registered tools. If the confidence is high (>0.9), it bypasses deliberation (search) and executes the tool immediately—mimicking the fast, intuitive action selection of the Basal Ganglia.

## Citations & Inspirations

1.  **Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J. (2017).** *The hippocampus as a predictive map.* Nature Neuroscience.
    *   *Inspiration*: The concept of organizing memory and context as a predictive map rather than a static store.

2.  **Vaswani, A., et al. (2017).** *Attention Is All You Need.* NeurIPS.
    *   *Implementation*: The `VectorNormalizer` utilizes a Transformer Encoder-Decoder architecture to attend to the input vector and project it into the normalized space.

3.  **Lewis, P., et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
    *   *Foundation*: The core RAG mechanism that CCP augments with neuro-symbolic routing and normalization.

## Features

### Task-Specific Training: Teach Your ChatBot New Domains
CCP includes a **Distillation Engine** that allows the system to learn new domains autonomously via "Web-Injection".

**How it works:**
1.  **Trigger**: You send a topic (e.g., "Quantum Computing") to the `/train_task` endpoint.
2.  **Distillation**: The engine searches the web, scrapes content, and uses an LLM teacher to generate high-quality "User-Query -> Ideal-Answer" pairs.
3.  **Fine-Tuning**: The `NeuralTrainer` loads these pairs and fine-tunes the `VectorNormalizer` for that specific domain.
4.  **Result**: A hot-swappable weight file (`weights/normalizer_quantum_computing.pt`) that "rotates" future queries about Quantum Computing into the optimal search space.

### Hybrid Inference: Privacy-First Architecture

CCP implements a **Hybrid Inference** model to balance privacy, cost, and performance:

1.  **Local Reasoning (Default)**: Routine reasoning, graph orchestration, and tool selection are handled by a **Local LLM (Qwen-2.5-1.5B)** running in a dedicated Docker container. This ensures that internal logic and structure remain private and cost-free.
2.  **Cloud Generation (Fallback)**: For heavy content generation or complex creative tasks, the system can seamlessly switch to the **Google Gemini API**.
3.  **Self-Healing Registry**: A specialized `FunctionRegistry` monitors your Python code (`src/ccp/functions/library/*.py`). On startup, it hashes your function source code and docstrings, automatically synchronizing them to the **Vector Memory (Qdrant)**. This ensures the LLM always has up-to-date semantic understanding of your tools without manual re-indexing.

### API Capabilities

-   **POST /chat**: Interact with the system, specifying `granularity_level` (0.0 - 1.0) and `task_domain` to load specific neural weights.
-   **POST /train_task**: Trigger the background learning process for new topics.