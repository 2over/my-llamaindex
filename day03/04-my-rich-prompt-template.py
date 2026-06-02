from llama_index.core.prompts import RichPromptTemplate

template = RichPromptTemplate(
    """
    {% chat role="system" %}
    你是多模态文档分析助手，需要结合图片内容和文本描述回答用户问题。
    核心规则:
    1.优先基于图片对应的文本描述分析信息
    2.若图片路径中包含"合同"关键词，重点关注文本中的金额、日期信息;
    3.回答需简介明了，分店说明关键信息
    {% endchat %}
    
    {% chat role="user" %}
    请分析一下图片和对应的文本信息，总结每份文件的核心内容:
    {% for img_path, text_content in multi_modal_data %}
    - 文件路径: {{ img_path }}
    - 文本描述: {{ text_content }}
    - 图片内容: {{ img_path | image}} # 标记为图片类型，供多模态模型解析
    {% endfor %}
    
    
    我的问题: 这些文件中是否包含合同类文件？ 如果有，核心信息是什么？
    {% endchat %}
    """
)
messages = template.format_messages(
    multi_modal_data=[
        ("contract_202403.png", "2024年3月采购合同：甲方为XX科技，乙方为YY制造，合同金额50万元，有效期1年"),
        ("contract_202403.png", "2024年Q1销售报告：总销售额1200万元，同比增长15%，覆盖3个省份"),
        ("invoice_202404.png", "2024年4月发票：金额8.5万元，对应项目为服务器采购")
    ]
)

print("=== 格式化后的多模态聊天消息列表 ===")
for idx, msg in enumerate(messages):
    print(f"\n【消息{idx + 1}】")
    print(f"角色：{msg.role}")
    print(f"内容: {msg.content.strip()}")