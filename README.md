# LLM-Generated Synthetic Contact Matrices


## Research Questions

### Main Analysis
*   **RQ1: How does the size of the dataset used for fine-tuning affect contact generation performance?**
    *   Method: Fine-tune models on stratified subsets (10%, 30%, 50%, and 80%) of the full MixIT questionnaire dataset.
    *   Evaluation: Compare generated data (starting from synthetic, real, and excluded populations) against observed data metrics.
*   **RQ2: Are there performance differences across different population groups?**
    *   Method: Stratify real and generated respondents based on key characteristics (age, gender, NUTS1 region, education, income, household size, occupation).
    *   Evaluation: Report distributions and contact matrices separately for each subgroup.

### Secondary Analysis
*   **RQ3: What impact does varying degrees of under-representation have on performance?**
    *   Method: Test generalization by progressively reintroducing data from a specific excluded subgroup (0%, 10%, 25%, 50%).
    *   Evaluation: Test the model specifically on the underrepresented group.

### Sensitivity Analysis
*   **RQ4: Do the results depend on the size of the language model used?**
    *   Method: Repeat the main analyses (RQ1/RQ2) using a larger parameter model from the same family (e.g., transitioning from 8B/9B to 27B/70B parameters).
    *   Evaluation: Compare error curves and biases to assess robustness.

## Data Pipeline

The project uses a structured ChatML/JSON format to maximize the LLM's understanding of semantic variables and ensure 100% parsing reliability during evaluation.

### Conditioning Variables (Respondent)
*   Participant Age
*   Participant Gender
*   Household Size
*   Attends Work
*   Attends School
*   Educational Attainment
*   Region (NUTS1)
*   Occupation
*   Household Monthly Net Income

### Generated Variables (Contacts)
*   Contact Age
*   Contact Gender
*   Contact Frequency
*   Physical Contact
*   Distance from Home
*   Relationship to Participant
*   Contact Setting
*   Location of Contact

## Repository Structure

```text
llm-contact-matrices-paper/
├── data/
│   ├── data_mixit_postpandemic_contacts/                 # Raw MixIT relational tables (ignored in git)
│   ├── processed/           # Formatted JSONL prompts/completions (ignored in git)
│   └── splits/              # Stratified train/test splits for RQ1 and RQ3
├── scripts/
│   ├── 01_prepare_data.py   # Merges raw tables and maps to natural language JSONL
├── notebooks/
├── .gitignore
└── README.md