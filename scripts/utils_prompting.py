import json

def build_demographic_string(demos):
    """Builds the standard input prompt for the LLM based on participant demographics."""
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
    Routes the data into the requested format for the prompt sweep.
    Returns a standard ChatML 'messages' array.
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
        age = demos.get('age')
        try:
            age_val = float(age)
            stage = "school-aged, meaning contacts will be highly assortative and concentrated in school settings" if age_val < 18 else "an adult, meaning contacts will likely distribute across work, home, and leisure"
        except (ValueError, TypeError):
            stage = "of unknown age"
            
        reasoning = (
            f"Thought process: The participant is {age} years old. They are {stage}. "
            f"Based on a household size of {demos.get('hh_size')} and an occupation of {demos.get('occupation')}, "
            f"I will generate {len(contacts)} contacts to reflect their daily mixing patterns.\n\n"
        )
        assistant_content = reasoning + json.dumps(contacts)
        
    elif prompt_style == "natural_language":
        # Format 3: Unstructured narrative text
        lines = [f"This participant recorded {len(contacts)} contacts today."]
        for i, c in enumerate(contacts, 1):
            lines.append(
                f"Contact {i} was a {c.get('Contact Age')}-year-old {c.get('Contact Gender')}. "
                f"They met at {c.get('Contact Setting')} ({c.get('Location of Contact')}). "
                f"They maintained a {c.get('Distance during Contact')} distance. "
                f"The relationship is {c.get('Relationship to Participant')} and they meet {c.get('Contact Frequency')}. "
                f"Physical contact: {c.get('Physical Contact')}."
            )
        assistant_content = "\n".join(lines)

    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]