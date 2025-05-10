from rest_framework import serializers

# Serializer for user-to-id mapping
class UserToIdMappingSerializer(serializers.Serializer):
    user_id = serializers.IntegerField()
    id = serializers.IntegerField()

# Serializer for item-to-id mapping
class ItemToIdMappingSerializer(serializers.Serializer):
    asin = serializers.CharField(max_length=255)
    id = serializers.IntegerField()

# Serializer for recommendations response
class RecommendationsSerializer(serializers.Serializer):
    user_id = serializers.IntegerField()
    recommended_items = serializers.ListField(
        child=serializers.CharField(max_length=255)
    )

# Serializer for the Graph data (You may adjust fields based on your graph structure)
class GraphDataSerializer(serializers.Serializer):
    graph = serializers.ListField(child=serializers.IntegerField())
    metadata = serializers.DictField()
