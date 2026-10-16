"""Runtime UI: the pre-match setup screen and the play / research renderer (FINAL-2 spec 15-25)."""


def bring_to_front() -> None:
    """Raise and focus the game window. On macOS a Python process started from a terminal opens its window behind
    the terminal app; SDL_RaiseWindow activates it. A no-op headless (dummy driver) or if SDL cannot."""
    try:
        import pygame
        if not pygame.display.get_init() or pygame.display.get_driver() == "dummy":
            return
        from pygame._sdl2.video import Window
        Window.from_display_module().focus()
    except Exception:  # noqa: BLE001 - focusing is a convenience, never a failure
        pass
