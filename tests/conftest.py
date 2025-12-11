import os

from hypothesis import settings


settings.register_profile("slow", max_examples=1000)
settings.register_profile("fast", max_examples=10)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
