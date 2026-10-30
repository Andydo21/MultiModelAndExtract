"""Plain service outcomes rendered by the HTTP adapters."""
from dataclasses import dataclass, field

@dataclass
class ServiceResult:
    payload: object
    status: int = 200
    content_type: str = 'application/json'
    headers: dict = field(default_factory=dict)
    template: str = ''
    redirect_to: str = ''

    def __setitem__(self, name, value):
        self.headers[name] = value
