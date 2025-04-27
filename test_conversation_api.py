import requests
import json
import sys

class ConversationApiTest:
    def __init__(self, base_url="http://localhost:9222", api_key="ragflow-Q3Njg2ODNjMTNjMDExZjBhYTE4MzU1Yz"):
        self.base_url = base_url
        self.api_key = api_key
        self.conversation_url = f"{base_url}/v1/conversation/ask"
        self.headers = {
            "Content-Type": "application/json"
        }
        self.results = []
    
    def test_conversation_streaming(self, message="你好，请介绍一下自己"):
        """
        Test the conversation API with streaming output - API key in payload
        """
        method_name = "API key in payload"
        payload = {
            "message": message,
            "stream": True,
            "api_key": self.api_key
        }
        
        print(f"Sending request to {self.conversation_url}")
        print(f"Request payload: {json.dumps(payload, ensure_ascii=False)}")
        
        result = self._make_request(self.conversation_url, headers=self.headers, payload=payload, method_name=method_name)
        self.results.append(result)
        return result

    def test_alternative_method(self, message="你好，请介绍一下自己"):
        """
        Test with API key as a query parameter
        """
        method_name = "API key as query parameter"
        params = {
            "api_key": self.api_key
        }
        
        payload = {
            "message": message,
            "stream": True
        }
        
        print(f"Sending request to {self.conversation_url} with API key as parameter")
        print(f"Request payload: {json.dumps(payload, ensure_ascii=False)}")
        
        result = self._make_request(self.conversation_url, headers=self.headers, payload=payload, 
                                  params=params, method_name=method_name)
        self.results.append(result)
        return result

    def test_x_api_key_header(self, message="你好，请介绍一下自己"):
        """
        Test with X-API-Key header
        """
        method_name = "X-API-Key header"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        
        payload = {
            "message": message,
            "stream": True
        }
        
        print(f"Sending request to {self.conversation_url} with X-API-Key header")
        print(f"Request payload: {json.dumps(payload, ensure_ascii=False)}")
        print(f"Headers: {headers}")
        
        result = self._make_request(self.conversation_url, headers=headers, payload=payload, method_name=method_name)
        self.results.append(result)
        return result

    def test_api_key_header(self, message="你好，请介绍一下自己"):
        """
        Test with api-key header
        """
        method_name = "api-key header"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "message": message,
            "stream": True
        }
        
        print(f"Sending request to {self.conversation_url} with api-key header")
        print(f"Request payload: {json.dumps(payload, ensure_ascii=False)}")
        print(f"Headers: {headers}")
        
        result = self._make_request(self.conversation_url, headers=headers, payload=payload, method_name=method_name)
        self.results.append(result)
        return result

    def test_url_path_method(self, message="你好，请介绍一下自己"):
        """
        Test with API key directly in the URL path
        """
        method_name = "API key in URL path"
        # Create URL with API key embedded in path
        url = f"{self.base_url}/v1/conversation/ask/{self.api_key}"
        
        payload = {
            "message": message,
            "stream": True
        }
        
        print(f"Sending request to {url} with API key in URL path")
        print(f"Request payload: {json.dumps(payload, ensure_ascii=False)}")
        
        result = self._make_request(url, headers=self.headers, payload=payload, method_name=method_name)
        self.results.append(result)
        return result

    def _make_request(self, url, headers, payload, params=None, method_name="Unknown method"):
        """
        Internal method to make the actual request and process the response
        """
        result = {
            "method": method_name,
            "url": url,
            "headers": headers,
            "payload": payload,
            "params": params,
            "status_code": None,
            "response": None,
            "success": False,
            "streaming_data": []
        }
        
        try:
            with requests.post(
                url, 
                headers=headers, 
                json=payload,
                params=params,
                stream=True
            ) as response:
                result["status_code"] = response.status_code
                
                print("\nStreaming response:")
                print("-" * 50)
                
                for chunk in response.iter_lines():
                    if chunk:
                        # Try to parse the chunk as JSON
                        try:
                            chunk_data = json.loads(chunk.decode('utf-8'))
                            # Print the content (adjust based on actual response format)
                            if 'content' in chunk_data:
                                print(chunk_data['content'], end='', flush=True)
                                result["streaming_data"].append(chunk_data['content'])
                            else:
                                print(f"Received data: {json.dumps(chunk_data, ensure_ascii=False)}")
                                result["response"] = chunk_data
                        except json.JSONDecodeError:
                            # If not JSON, print the raw chunk
                            decoded = chunk.decode('utf-8')
                            print(f"Received raw chunk: {decoded}")
                            result["streaming_data"].append(decoded)
                
                print("\n" + "-" * 50)
                
                # If the response was successful (200), mark it as such
                # Check for known error response codes and ensure we don't treat them as success
                if (response.status_code == 200 and 
                    not result.get("response", {}).get("code") == 401 and
                    not result.get("response", {}).get("code") == 100 and
                    not ("<NotFound '404: Not Found'>" in str(result.get("response", {}).get("message", "")))):
                    result["success"] = True
                    
        except requests.RequestException as e:
            print(f"Error during API call: {e}")
            result["error"] = str(e)
            
            if hasattr(e, 'response') and e.response:
                result["status_code"] = e.response.status_code
                print(f"Response status code: {e.response.status_code}")
                
                try:
                    result["response"] = e.response.json()
                except:
                    result["response"] = e.response.text
                    
                print(f"Response body: {result['response']}")
                
        return result

    def summarize_results(self):
        """
        Print a summary of all the test results
        """
        print("\n" + "=" * 80)
        print("SUMMARY OF AUTHENTICATION METHODS TESTED")
        print("=" * 80)
        
        print(f"{'Method':<25} | {'Status':<10} | {'Response Code':<15} | {'Result':<30}")
        print("-" * 80)
        
        for result in self.results:
            status = "SUCCESS" if result["success"] else "FAILED"
            
            # Extract the response code from the result
            response_code = "Unknown"
            if result.get("response") and isinstance(result["response"], dict):
                response_code = result["response"].get("code", "N/A")
            elif result.get("status_code"):
                response_code = result["status_code"]
                
            print(f"{result['method']:<25} | {status:<10} | {response_code!s:<15} | {self._get_result_summary(result):<30}")
            
        print("=" * 80)
        
        # If all methods failed, provide a suggestion
        if not any(result["success"] for result in self.results):
            print("\nAll authentication methods failed. Possible reasons:")
            print("1. The API key may be incorrect or expired")
            print("2. The API may be temporarily unavailable")
            print("3. The API may use a different authentication method not tested")
            print("4. The server may be running but the API endpoint may be different")
            print("\nSuggestions:")
            print("- Double-check the API key")
            print("- Consult the API documentation at http://localhost:9222/user-setting/api")
            print("- Try using a tool like curl with verbose mode to debug further")
    
    def _get_result_summary(self, result):
        """
        Get a short summary of the result
        """
        if result["success"]:
            if result["streaming_data"]:
                return f"Received {len(result['streaming_data'])} chunks"
            return "Success"
        
        if isinstance(result.get("response"), dict):
            message = result["response"].get("message", "")
            if message:
                return message[:30] + "..." if len(message) > 30 else message
        
        if result.get("error"):
            return result["error"][:30] + "..." if len(result["error"]) > 30 else result["error"]
            
        return "Failed with unknown reason"

def run_all_tests():
    """Run all authentication test methods"""
    api_test = ConversationApiTest()
    
    # Method 1: API key in JSON payload
    print("METHOD 1: API key in payload")
    api_test.test_conversation_streaming("你好，请介绍一下自己")
    
    print("\n" + "=" * 50 + "\n")
    
    # Method 2: API key as query parameter
    print("METHOD 2: API key as query parameter")
    api_test.test_alternative_method("请解释什么是大语言模型")
    
    print("\n" + "=" * 50 + "\n")
    
    # Method 3: X-API-Key header
    print("METHOD 3: X-API-Key header")
    api_test.test_x_api_key_header("What is RAGFlow?")
    
    print("\n" + "=" * 50 + "\n")
    
    # Method 4: api-key header
    print("METHOD 4: api-key header")
    api_test.test_api_key_header("Tell me about yourself")
    
    print("\n" + "=" * 50 + "\n")
    
    # Method 5: API key in URL path
    print("METHOD 5: API key in URL path")
    api_test.test_url_path_method("What can you do?")
    
    # Print summary
    api_test.summarize_results()

if __name__ == "__main__":
    run_all_tests() 