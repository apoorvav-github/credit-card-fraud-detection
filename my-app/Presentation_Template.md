### Deep Dive into the Influence of Design Choices on Model Performance in Federated Learning 

**A Presentation for [Your Audience]** 

**Date:** August 3, 2025 

**Presenter:** [Your Name/Team Name] 

--- 

### Slide 1: Title Slide 

**Deep Dive into the Influence of Design Choices on Model Performance in Federated Learning** 

**Goal:** To understand the influence of different design choices under IID and non-IID conditions to facilitate easier decision-making. 

**Keywords:** Federated Learning, Model Performance, IID, non-IID, Communication Architecture, Aggregation Strategy 

--- 

### Slide 2: Agenda 

*   Introduction to Federated Learning (FL) 
*   Key Design Choices in FL 
    *   Communication Architecture 
    *   Aggregation Strategy 
*   The Challenge of Data Heterogeneity (IID vs. non-IID) 
*   Experimental Setup 
    *   Benchmark Dataset 
    *   Reference Scenario 
    *   Scenario Development 
*   Evaluation Methodology 
    *   Local vs. Global Evaluation 
*   Implementation Highlights with Flower 
*   Expected Results & Discussion 
*   Add-on: Impact of Client Scale 
*   Conclusion & Future Work 

--- 

### Slide 3: Introduction to Federated Learning (FL) 

*   **What is it?** A machine learning setting where multiple clients collaboratively train a model, coordinated by a central server, while keeping the training data decentralized. 
*   **Why is it important?** 
    *   **Privacy:** Raw data never leaves the client devices. 
    *   **Reduced Communication Costs:** Only model updates, not the entire dataset, are sent to the server. 
    *   **Real-time Learning:** Enables on-device model training and predictions. 
*   **Core Idea:** Train locally, aggregate globally. Local models are trained on individual client data, and their updates are aggregated to create a more robust global model. 

--- 

### Slide 4: Key Design Choices: Communication Architecture 

*   **Centralized:** A central server orchestrates the entire process, from client selection to model aggregation. This is the most common architecture. 
    *   *Pros:* Simplicity, direct control over the learning process. 
    *   *Cons:* Single point of failure, potential communication bottleneck. 
*   **Decentralized:** Clients communicate directly with each other in a peer-to-peer fashion, without a central server. 
    *   *Pros:* Increased robustness, no single point of failure. 
    *   *Cons:* More complex coordination, potential for slower convergence due to indirect communication. 
*   **Cyclic/Hierarchical:** A hybrid approach where clients are organized into clusters, with a server for each cluster, and a higher-level server aggregating models from the cluster servers. This can be beneficial for large-scale IoT applications. 

--- 

### Slide 5: Key Design Choices: Aggregation Strategy 

*   **Synchronous:** The server waits for updates from all selected clients before aggregating them to update the global model. Federated Averaging (FedAvg) is a popular synchronous algorithm. 
    *   *Pros:* Simple to implement, stable convergence on IID data. 
    *   *Cons:* Suffers from the "straggler" problem, where the entire process is slowed down by the slowest client. 
*   **Asynchronous:** The server updates the global model as soon as it receives an update from any client, without waiting for others. 
    *   *Pros:* Improved efficiency and resource utilization, robust to client dropouts. 
    *   *Cons:* Can suffer from "stale" model updates, potentially leading to slower convergence or instability, especially with non-IID data. 
*   **Hybrid (Semi-asynchronous):** A middle ground where the server waits for a certain number of clients or for a specific time before aggregating updates. This aims to balance the trade-offs between synchronous and asynchronous approaches. 

--- 

### Slide 6: The Challenge of Data Heterogeneity 

*   **IID (Independent and Identically Distributed):** Data across all clients is drawn from the same underlying distribution. This is the ideal, but often unrealistic, scenario. 
*   **Non-IID (Non-Independent and Identically Distributed):** The data distribution varies across clients. This is a significant challenge in real-world FL applications. 
    *   **Impact on Performance:** 
        *   **Reduced Accuracy:** The global model may struggle to generalize across all clients. 
        *   **Slower Convergence:** Conflicting updates from clients with different data distributions can slow down the training process. 
        *   **Unfairness:** The model may perform poorly for clients with under-represented data. 
*   **Our Focus:** Investigating the impact of design choices under both IID and various degrees of non-IID (label distribution skew) conditions. 

--- 

### Slide 7: Experimental Setup: Benchmark Dataset 

*   **Task:** Credit Card Fraud Detection or Credit Score Classification. These are relevant financial applications where data privacy is paramount. 
*   **Potential Datasets:** 
    *   **Credit Card Fraud Detection:** 
        *   Publicly available datasets on platforms like Kaggle. It's crucial to address the high class imbalance often present in these datasets. 
        *   Some research utilizes datasets like the 2018CN and 2023EU datasets. 
    *   **Credit Scoring:** 
        *   "Give Me Some Credit" dataset from Kaggle. 
        *   Freddie Mac's Single-Family Loan-Level Dataset for mortgage risk. 

*   **Data Preparation:** The chosen dataset will be partitioned among clients to simulate both IID and non-IID scenarios. 

--- 

### Slide 8: Experimental Setup: Reference Scenario 

*   **Objective:** To establish a baseline for comparison against our design modifications. 
*   **Configuration:** 
    *   **Communication Architecture:** Centralized. 
    *   **Aggregation Strategy:** Synchronous (e.g., FedAvg). 
    *   **Data Distribution:** IID. 
    *   **Number of Clients:** A manageable number (e.g., 3-10) to allow for realistic benchmarks while keeping computational requirements low. 
*   **Rationale:** This setup represents a standard and well-understood FL implementation, providing a solid foundation for evaluating the impact of changes. 

--- 

### Slide 9: Experimental Setup: Scenario Development 

*   We will systematically vary the aggregation strategy and data heterogeneity to create a matrix of experimental scenarios. 
*   **Example Scenarios (with 3 clients):** 
    *   **Aggregation Strategy:** Synchronous, Asynchronous, Hybrid. 
    *   **Data Heterogeneity:** 
        *   IID 
        *   Non-IID (weak, medium, and strong label distribution skew). 

This structured approach will allow us to isolate and quantify the influence of each design choice. 

--- 

### Slide 10: Evaluation Methodology 

We will employ a two-pronged evaluation approach: 

1.  **Isolated Training:** A local model is trained for each client solely on its own dataset. This serves as a benchmark for the performance of a non-collaborative approach. 

2.  **Federated Training:** Local models are aggregated to create a global model. 

*   **Local Evaluation:** 
    *   **What:** Compare the performance of the isolated model and the global model on the local test data of *each client*. 
    *   **Why:** To assess how well the global model generalizes to the specific data distribution of each individual client. This is crucial for understanding fairness and personalization. 
*   **Global Evaluation:** 
    *   **What:** Evaluate the performance of the final global model on a centralized, IID test dataset. 
    *   **Why:** To get a comprehensive measure of the overall performance of the collaboratively trained model. 

--- 

### Slide 11: Implementation Highlights with Flower 

*   **Flower Framework:** We will leverage the Flower framework for our implementation. 
*   **Why Flower?** 
    *   **Flexibility:** It is framework-agnostic, supporting frameworks like PyTorch and TensorFlow. 
    *   **Ease of Use:** Provides a high-level API that simplifies the implementation of federated learning systems. 
    *   **Scalability:** Designed to handle a large number of clients. 
    *   **Customization:** Allows for easy implementation of different communication architectures and aggregation strategies. 
*   **Our Implementation:** We will define custom strategies in Flower to implement synchronous, asynchronous, and hybrid aggregation. 

--- 

### Slide 12: Expected Results & Discussion 

*   **IID Conditions:** We expect synchronous and hybrid aggregation strategies to perform well, likely outperforming a purely asynchronous approach in terms of final model accuracy. 
*   **Non-IID Conditions:** 
    *   We anticipate a performance degradation for all strategies as the degree of non-IID skew increases. 
    *   Asynchronous and hybrid strategies might show resilience to stragglers, but their final accuracy could be lower than synchronous methods due to stale updates. 
    *   We will analyze the trade-off between convergence speed (wall-clock time) and final model performance. 
*   **Local vs. Global Performance:** We expect the global model to outperform the isolated local models, especially for clients with smaller or less diverse datasets. However, under strong non-IID, the global model might perform worse on some local test sets than the specialized local model. 

--- 

### Slide 13: Add-on: Impact of Client Scale 

*   **Investigation:** We will conduct an additional set of experiments to analyze the influence of the number of clients participating in the training process. 
*   **Hypothesis:** 
    *   Increasing the number of clients with IID data is likely to improve the global model's performance. 
    *   With non-IID data, a larger number of clients could exacerbate the "client drift" problem, where local models diverge significantly. 
*   **Evaluation:** We will track local and global performance metrics as we vary the number of clients to understand this dynamic. 

--- 

### Slide 14: Conclusion & Future Work 

*   **Key Takeaways:** This research will provide a systematic comparison of key design choices in Federated Learning, offering insights into their impact on model performance under varying data conditions. 
*   **Decision-Making:** The results will help practitioners make informed decisions about the most suitable communication architecture and aggregation strategy for their specific use case and data landscape. 
*   **Future Work:** 
    *   Explore more advanced aggregation strategies designed to mitigate the effects of non-IID data. 
    *   Investigate the impact of other forms of data heterogeneity, such as feature distribution skew. 
    *   Incorporate privacy-preserving techniques like differential privacy and secure aggregation into the framework. 

--- 

### Slide 15: Q&A and Thank You! 

**[Your Name/Team Name]** 

**[Your Contact Information]**
