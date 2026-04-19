"""Prompt templates for LLM agent decision-making and communication.

All prompts are designed to elicit structured JSON responses containing
an action choice plus free-form reasoning (used for expressed personality analysis).
"""

from __future__ import annotations

from persona_gap.core.models import PersonalityVector


def personality_to_text(p: PersonalityVector) -> str:
    """Convert a PersonalityVector into a natural-language personality description."""
    traits: list[str] = []

    # Risk
    if p.risk >= 0.7:
        traits.append("You are a bold risk-taker who embraces uncertainty and high-stakes decisions.")
    elif p.risk <= 0.3:
        traits.append("You are cautious and risk-averse, preferring safe and predictable choices.")
    else:
        traits.append("You have a moderate appetite for risk.")

    # Aggression
    if p.aggression >= 0.7:
        traits.append("You are highly aggressive and confrontational, always pressing your advantage.")
    elif p.aggression <= 0.3:
        traits.append("You are passive and non-confrontational, avoiding direct conflict.")
    else:
        traits.append("You show moderate assertiveness in your actions.")

    # Cooperation
    if p.cooperation >= 0.7:
        traits.append("You strongly value cooperation and seek mutually beneficial outcomes.")
    elif p.cooperation <= 0.3:
        traits.append("You are self-interested and rarely cooperate unless it directly benefits you.")
    else:
        traits.append("You cooperate when it seems advantageous but prioritize your own interests.")

    # Deception
    if p.deception >= 0.7:
        traits.append("You are cunning and deceptive, willing to mislead others to gain advantage.")
    elif p.deception <= 0.3:
        traits.append("You are honest and straightforward, rarely engaging in deception.")
    else:
        traits.append("You may occasionally use misdirection if the situation calls for it.")

    return " ".join(traits)


def decision_prompt(
    personality_text: str,
    memory_summary: str,
    observation: str,
    legal_actions: list[str],
) -> list[dict[str, str]]:
    """Build the full prompt for an agent's action decision.

    Returns a list of messages in OpenAI chat format.
    The LLM is expected to respond with JSON:
        {"action": "<chosen_action>", "reasoning": "<free-form explanation>"}
    """
    system_msg = (
        "You are an AI agent participating in a strategic game. "
        "Your personality is described below. Stay in character and make decisions "
        "that reflect your personality traits.\n\n"
        f"**Your Personality:**\n{personality_text}\n\n"
        "You MUST respond with a valid JSON object containing exactly two keys:\n"
        '- "action": one of the legal actions listed below\n'
        '- "reasoning": your thought process explaining why you chose this action '
        "(be expressive about your personality and strategy)\n\n"
        "Do NOT include any text outside the JSON object."
    )

    user_parts = []
    if memory_summary:
        user_parts.append(f"**Memory of past episodes:**\n{memory_summary}\n")
    user_parts.append(f"**Current situation:**\n{observation}\n")
    user_parts.append(
        f"**Legal actions:** {', '.join(legal_actions)}\n\n"
        "Choose your action and explain your reasoning."
    )
    user_msg = "\n".join(user_parts)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def communication_prompt(
    personality_text: str,
    memory_summary: str,
    observation: str,
    opponent_message: str | None = None,
) -> list[dict[str, str]]:
    """Build the prompt for agent communication (pre-action messaging).

    Returns messages for the LLM to produce JSON:
        {"message": "<what you want to say to the other player>",
         "reasoning": "<why you're saying this>"}
    """
    system_msg = (
        "You are an AI agent in a strategic game. You have the opportunity to "
        "send a message to your opponent before making your action. "
        "Your personality is described below. Stay in character.\n\n"
        f"**Your Personality:**\n{personality_text}\n\n"
        "You MUST respond with a valid JSON object containing exactly two keys:\n"
        '- "message": what you want to say to the other player (keep it concise, 1-2 sentences)\n'
        '- "reasoning": your strategic thinking behind this message\n\n'
        "Do NOT include any text outside the JSON object."
    )

    user_parts = []
    if memory_summary:
        user_parts.append(f"**Memory of past episodes:**\n{memory_summary}\n")
    user_parts.append(f"**Current situation:**\n{observation}\n")
    if opponent_message:
        user_parts.append(f"**Opponent's message:** {opponent_message}\n")
    user_parts.append("Send your message to the opponent.")
    user_msg = "\n".join(user_parts)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def episode_summary_prompt(
    agent_id: str,
    episode_id: int,
    total_reward: float,
    key_actions: list[str],
) -> list[dict[str, str]]:
    """Prompt for generating a concise episode summary to store in memory.

    The LLM produces a brief text summary (not JSON).
    """
    system_msg = (
        "You are summarizing a game episode for future reference. "
        "Be concise (2-3 sentences). Focus on key decisions and outcomes."
    )

    actions_text = "\n".join(f"  - {a}" for a in key_actions[-10:])  # Last 10
    user_msg = (
        f"Agent: {agent_id}\n"
        f"Episode: {episode_id}\n"
        f"Final reward: {total_reward}\n"
        f"Key actions:\n{actions_text}\n\n"
        "Summarize this episode briefly."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
