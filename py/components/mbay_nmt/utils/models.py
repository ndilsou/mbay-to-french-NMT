from bson import ObjectId as _ObjectId
import datetime as dt
from typing import Annotated, Any, Callable
import uuid
from pydantic import (
    AfterValidator,
    ConfigDict,
    UUID4,
    BaseModel as PydanticBaseModel,
    Field,
    GetJsonSchemaHandler,
    field_serializer,
)
from pydantic.alias_generators import to_camel
from pydantic_core import core_schema
from pydantic.json_schema import JsonSchemaValue


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


def uuid_field() -> Any:
    return Field(default_factory=uuid.uuid4)


def _serialize_dt(dt: dt.datetime):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def new_object_id():
    return ObjectId()


class _ObjectIdTypeAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        """
        We return a pydantic_core.CoreSchema that behaves in the following ways:

        * ints will be parsed as `ThirdPartyType` instances with the int as the x attribute
        * `ThirdPartyType` instances will be parsed as `ThirdPartyType` instances without any changes
        * Nothing else will pass validation
        * Serialization will always return just an int
        """

        def validate_from_str(value: str) -> _ObjectId:
            return _ObjectId(value)

        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(validate_from_str),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(_ObjectId),
                    from_str_schema,
                ]
            ),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: str(instance)
            # ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Use the same schema that would be used for `str`
        return handler(core_schema.str_schema())


ObjectId = Annotated[_ObjectId, _ObjectIdTypeAnnotation]

# def check_object_id(value: str) -> str:
#     if not _ObjectId.is_valid(value):
#         raise ValueError("Invalid ObjectId")
#     return value


# ObjectId = Annotated[str, AfterValidator(check_object_id)]


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(
        populate_by_name=True, extra="allow", alias_generator=to_camel
    )


class Entity(BaseModel):
    id: ObjectId = Field(..., default_factory=new_object_id, alias="_id")
    created_at: dt.datetime = Field(default_factory=utcnow)
    updated_at: dt.datetime = Field(default_factory=utcnow)

    model_config = ConfigDict(validate_assignment=True, alias_generator=to_camel)

    @field_serializer("created_at", "updated_at")
    def _serialize_dt(self, dt: dt.datetime, _info: Any):
        return _serialize_dt(dt)

    def refresh_updated_at(self):
        self.updated_at = utcnow()


class ValueObject(BaseModel):
    model_config = ConfigDict(frozen=True, alias_generator=to_camel)
