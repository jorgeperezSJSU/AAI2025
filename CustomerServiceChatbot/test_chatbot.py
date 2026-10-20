"""Offline tests use scripted model doubles, never claimed as live API outputs."""
import unittest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from chatbot import (RouterAgent, SpecialistAgent, Route, ask, get_order_details,
                     get_laptop_price, recommend_laptops, get_refund_policy)

class FakeRouter:
    def invoke(self, messages):
        text = messages[-1].content.lower()
        route = 'REFUND' if 'return' in text else 'ORDER' if 'order' in text else 'PRODUCT'
        return Route(destination=route)

class FakeBound:
    def __init__(self, tools):
        self.tool = tools[0].name
    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content='OFFLINE TEST DOUBLE: ' + messages[-1].content)
        args = {'max_budget': 1500} if self.tool == 'recommend_laptops' else (
            {'order_id': 'ORD-7311'} if self.tool == 'get_order_details' else {})
        return AIMessage(content='', tool_calls=[{'name': self.tool, 'args': args, 'id': 'test-call'}])

class FakeModel:
    def with_structured_output(self, schema):
        return FakeRouter()
    def bind_tools(self, tools):
        return FakeBound(tools)

class ChatbotTests(unittest.TestCase):
    def test_catalog_and_missing_order(self):
        self.assertEqual(len(recommend_laptops.invoke({'max_budget':1500})['matches']),2)
        self.assertEqual(recommend_laptops.invoke({'max_budget':100})['matches'],[])
        self.assertEqual(get_laptop_price.invoke({'laptop_name':'alpha'})['matches'][0]['price_usd'],1499)
        self.assertIn('error',get_order_details.invoke({'order_id':'ORD-9999'}))
        self.assertEqual(get_order_details.invoke({'order_id':'ORD-7311'})['quantity'],2)

    def test_routes_memory_and_isolation(self):
        graph=RouterAgent(FakeModel()).router_graph
        for turn,(q,expected) in enumerate([('Laptop under 1500','PRODUCT'),('How much is it?','PRODUCT'),
                    ('Can I return it?','REFUND'),('Order ORD-7311','ORDER')],1):
            state=ask(graph,q,'test-session')
            self.assertEqual(state['destination'],expected)
            self.assertEqual(sum(isinstance(m,HumanMessage) for m in state['messages']),turn)
            self.assertTrue(state['tools_used'])
        fresh=ask(graph,'Hello','other-session')
        self.assertEqual(sum(isinstance(m,HumanMessage) for m in fresh['messages']),1)
        self.assertEqual(len(graph.get_state({'configurable':{'thread_id':'test-session'}}).values['messages']),16)

    def test_multiple_tool_calls(self):
        agent=SpecialistAgent(FakeModel(),[get_laptop_price,get_refund_policy],'Test')
        call=AIMessage(content='',tool_calls=[
            {'name':'get_laptop_price','args':{'laptop_name':'Gamma'},'id':'one'},
            {'name':'get_refund_policy','args':{},'id':'two'}])
        results=agent.call_tools({'messages':[call]})['messages']
        self.assertEqual([m.tool_call_id for m in results],['one','two'])

if __name__=='__main__':
    unittest.main(verbosity=2)
