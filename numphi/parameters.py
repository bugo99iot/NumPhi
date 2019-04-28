
# define types of influence
# "drag_down": tolerance of the influenced will descrese if tolerance of influencer is lower
# "drag_up": tolerance of the influenced will inscrese if tolerance of influencer is higer
# "always": both the above
# "none": do not influence

INFLUENCE_OPTIONS = ["drag_down", "drag_up", "always"]

# define types of reinforcement
# "when_tolerant": attack and defence of influenced will change if average of tolerances > 0.5
# "when_intolerant": attack and defence of influenced will change if average of tolerances < 0.5
# "always": both the above
# "none": don't reinforce

REINFORCE_OPTIONS = ["when_intolerant", "when_tolerant", "always"]
