"""Schedule expensive observation checks; never remove denoiser evaluations."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CheckSkipping:
    enabled: bool = False
    automatic: bool = False
    skip_count: int = 1

    def validate(self):
        if isinstance(self.skip_count, bool) or not isinstance(self.skip_count, int) or not 1 <= self.skip_count <= 8:
            raise ValueError('checks_to_skip must be an integer from 1 to 8')
        return self

    def resolve(self, settings, interval):
        self.validate()
        if not self.enabled:
            return settings, interval
        if self.automatic:
            return replace(settings, schedule='adaptive', max_interval=self.skip_count + 1,
                           detailed_checks=True), 1
        return replace(settings, schedule='fixed'), self.skip_count + 1
