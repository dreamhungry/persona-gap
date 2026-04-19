# How to add action annotations for a new game / environment
#
# 1. Create a new TOML file in this directory, e.g. `uno.toml`
#
# 2. Define an `[actions.<action_name>]` section for each possible action
#    in the game. Each action has 4 boolean personality dimensions:
#
#    [actions.draw_four_wild]
#    is_risky = true
#    is_aggressive = true
#    is_cooperative = false
#    is_deceptive = true
#
# 3. Update your experiment TOML to point to the new file:
#
#    [env]
#    adapter = "rlcard"
#    game_name = "uno"
#    action_annotations_path = "configs/envs/uno.toml"
#
# Tips:
# - Not every action needs to be tagged on every dimension.
#   Default is `false` for all if omitted.
# - An action can be tagged on multiple dimensions simultaneously
#   (e.g., bluffing in poker is both risky and deceptive).
