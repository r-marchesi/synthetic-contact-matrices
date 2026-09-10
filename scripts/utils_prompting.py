import json

def build_demographic_string(demos):
    """Builds the standard input prompt for the LLM."""
    return (
        f"Generate the synthetic epidemiological social contacts for a participant "
        f"with the following demographics:\n"
        f"- Participant Age: {demos.get('age')}\n"
        f"- Participant Gender: {demos.get('gender')}\n"
        f"- Household Size: {demos.get('hh_size')}\n"
        f"- Occupation: {demos.get('occupation', 'Unknown')}"
    )

def format_conversation(demos, contacts, prompt_style="json_strict"):
    """
    Routes the data into the requested format (Axis A of the Grid Search).
    Returns a standard HuggingFace 'messages' array.
    """
    user_content = build_demographic_string(demos)
    
    if prompt_style == "json_strict":
        # The Baseline: Strict JSON array of objects
        assistant_content = json.dumps(contacts)

    elif prompt_style == "yaml_style":
        # Format 1: Semi-structured, lower token overhead
        lines = []
        for c in contacts:
            lines.append("- contact:")
            for k, v in c.items():
                lines.append(f"    {k}: {v}")
        assistant_content = "\n".join(lines) if lines else "[]"

    elif prompt_style == "chain_of_thought":
        # Format 2: Forces the model to reason before outputting JSON
        reasoning = f"Thought process: The participant is {demos.get('age')} years old. "
        if int(demos.get('age', 0)) < 18:
            reasoning += "They are school-aged, so I should generate highly assortative school contacts. "
        else:
            reasoning += "They are an adult, so I should balance home, work, and leisure contacts. "
            
        reasoning += f"Generating {len(contacts)} contacts based on these traits.\n\n"
        assistant_content = reasoning + json.dumps(contacts)
        
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]