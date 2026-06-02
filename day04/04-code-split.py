from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import CodeSplitter
from tree_sitter_language_pack import get_parser


class NodeAdapter:
    def __init__(self, node):
        self.node = node

    @property
    def children(self):
        return [NodeAdapter(self.node.child(i)) for i in range(self.node.child_count())]

    @property
    def start_byte(self):
        return self.node.start_byte()

    @property
    def end_byte(self):
        return self.node.end_byte()

    @property
    def type(self):
        return self.node.kind()


class TreeAdapter:
    def __init__(self, tree):
        self.tree = tree

    @property
    def root_node(self):
        return NodeAdapter(self.tree.root_node())


class ParserAdapter:
    def __init__(self, parser):
        self.parser = parser

    def parse(self, source):
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        return TreeAdapter(self.parser.parse(source))


# 读取文件
documents = SimpleDirectoryReader(input_files=["./03-md-element-parser.py"]).load_data()

# 初始化代码分割器
splitter = CodeSplitter(
    language="python",
    chunk_lines=50,  # 每块行数
    chunk_lines_overlap=10,  # 重叠的行数
    max_chars=300,  # 块最大的字符数量
    parser=ParserAdapter(get_parser("python")),
)

# 将文档转换成节点
nodes = splitter.get_nodes_from_documents(documents)
for node in nodes:
    print(f"Type: {node.metadata}\n Text: {node.text}\n{'=' * 50}")
