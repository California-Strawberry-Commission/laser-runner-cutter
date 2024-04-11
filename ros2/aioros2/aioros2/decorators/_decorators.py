class RosDefinition:
    pass


def idl_to_kwargs(req):
        msg_keys = req.get_fields_and_field_types().keys()
        return {k: getattr(req, k) for k in msg_keys}