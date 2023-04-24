from src.providers import GenerativeModel
from src.model import DialogModel, SingleTurnDialog

def _together_format(dialogs: DialogModel) -> str:
    """Format the dialog to the format that the API expects
    Together expects the dialog to be in the following format:
    <human>: text\n<bot>: text\n<human>: text\n<bot>: text\n
    """
    def _format_single_turn(dialog: SingleTurnDialog) -> str:
        if dialog.role == "USER":
            return f"<human>: {dialog.text}"
        if dialog.role == "ASSISTANT":
            return f"<bot>: {dialog.text}"
    previous_prompt= "\n".join([_format_single_turn(dialog) for dialog in dialogs.dialogs])
    return f"{previous_prompt}\n<bot>:"

def _anthropic_format(dialogs: DialogModel) -> str:
    """Format the dialog to the format that the API expects
    Anthropic expects the dialog to be in the following format:
    \n\nHuman: text\n\nAssistant: text\n\nHuman: text\n\nAssistant: text
    """
    def _format_single_turn(dialog: SingleTurnDialog) -> str:
        if dialog.role == "USER":
            return f"\n\nHuman: {dialog.text}"
        if dialog.role == "ASSISTANT":
            return f"\n\nAssistant: {dialog.text}"
    previous_prompt = "".join([_format_single_turn(dialog) for dialog in dialogs.dialogs])
    return f"{previous_prompt}\n\nAssistant:"

async def chat_shortcut(model: GenerativeModel, dialogs: DialogModel):
    payload = dialogs.body
    if model.prefix == 'together':
        payload['prompt'] = _together_format(dialogs)
    elif model.prefix == 'anthropic':
        payload['prompt'] = _anthropic_format(dialogs)
    else:
        raise ValueError(f"Unknown model prefix: {model.prefix}")
    return await model(payload)