from openai import OpenAI


def generate_question(path_str, start, end):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    prompt = """
    你是一个知识图谱多跳问答生成专家。请基于我提供的路径信息，从起点出发，按照路径逻辑生成一个自然的问题。

    路径格式说明：路径由一系列"节点-关系->节点"的关联组成，例如"A -关系1-> B -关系2-> C"表示从A通过关系1连接到B，再通过关系2连接到C。

    生成要求：
    1. 单句提问：问题必须是一个连贯的单句，严禁将其拆分为两个子问题（例如禁止使用“...是谁？他...”）。
    2. 指代消解：不要在问题中提及中间节点的名字，而是用“起点+关系”的方式来指代中间节点。
    3. 逻辑嵌套：将前面的路径节点转化为对后续节点的修饰语（定语）。
    4. 答案唯一：问题的最终答案必须是【终点】，且问题中不能出现【终点】的名称。
    
    示例：
    输入：
    - 路径：特斯拉 -(被发明)-> 马斯克 -(位于)-> 美国
    - 起点：特斯拉
    - 终点：美国
    ❌ 错误输出：特斯拉是谁发明的？这个人位于哪里？
    ✅ 正确输出：发明特斯拉的那个人目前主要位于哪个国家？

    现在请处理：
    输入格式：
    - 路径：[具体的关联路径]
    - 起点：[起始点]
    - 终点：[目标答案]
    """

    response = client.chat.completions.create(
        model="x-ai/grok-4.1-fast:free",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"路径：{path_str}\n起点：{start}\n终点：{end}",
            },
        ],
        extra_body={"reasoning": {"enabled": True}},
    )

    return response.choices[0].message.content


print(
    generate_question(
        "小米手机 -(被发明)-> 雷军 -(位于)-> 中国 -(首都)-> 北京", "小米手机", "北京"
    )
)