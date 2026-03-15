import asyncio
import time
from rich.console import Console
from rich.tree import Tree
from rich.live import Live
from rich.panel import Panel

def build_agent_tree(session_id: str, agents_data: list):
    tree = Tree(f"[bold cyan]Session: {session_id}[/bold cyan]")
    
    agent_nodes = {}
    
    # Simple hierarchy mapping for testing
    # Each item: (id, parent_id, status)
    for agent_id, parent_id, status in agents_data:
        label = f"Agent: [bold yellow]{agent_id}[/bold yellow] ([dim]{status}[/dim])"
        if parent_id is None:
            node = tree.add(label)
            agent_nodes[agent_id] = node
        else:
            parent_node = agent_nodes.get(parent_id, tree)
            node = parent_node.add(label)
            agent_nodes[agent_id] = node
            
    return tree

async def main():
    console = Console()
    session_id = "test_hierarchy_001"
    
    # Mock data that evolves
    agents_data = [
        ("supervisor", None, "Planning"),
    ]
    
    console.print(Panel.fit("[bold blue]Testing Hierarchical UI[/bold blue]"))
    
    with Live(build_agent_tree(session_id, agents_data), console=console, refresh_per_second=2) as live:
        await asyncio.sleep(1)
        
        # 1. Supervisor starts working
        agents_data[0] = ("supervisor", None, "Decomposing")
        live.update(build_agent_tree(session_id, agents_data))
        await asyncio.sleep(1)
        
        # 2. Sub-agent 1 spawned
        agents_data.append(("researcher_1", "supervisor", "Initializing"))
        live.update(build_agent_tree(session_id, agents_data))
        await asyncio.sleep(1)
        
        # 3. Sub-agent 2 spawned
        agents_data.append(("researcher_2", "supervisor", "Initializing"))
        live.update(build_agent_tree(session_id, agents_data))
        await asyncio.sleep(1)
        
        # 4. Researcher 1 spawns a tool analyst (Depth 2)
        agents_data.append(("tool_analyst", "researcher_1", "Searching"))
        agents_data[1] = ("researcher_1", "supervisor", "Awaiting Child")
        live.update(build_agent_tree(session_id, agents_data))
        await asyncio.sleep(2)
        
        # 5. Final state
        agents_data[3] = ("tool_analyst", "researcher_1", "COMPLETED")
        agents_data[1] = ("researcher_1", "supervisor", "Reporting")
        live.update(build_agent_tree(session_id, agents_data))
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
