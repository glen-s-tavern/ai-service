import React, { useState } from 'react';
import { Tabs, Input, Card, List, Tag, Typography } from 'antd';
import { SearchOutlined, CalendarOutlined, BankOutlined, DollarOutlined } from '@ant-design/icons';
import TenderDetails from './components/TenderDetails';
import { mockTenders } from './mockData';

const { Text } = Typography;

interface Tab {
  id: string;
  query: string;
  tenders: Tender[];
}

interface Tender {
  purchase_number: string;
  tender_name: string;
  deadline?: string;
  amount?: string;
  organization?: string;
}

const App: React.FC = () => {
  const [tabs, setTabs] = useState<Tab[]>([{
    id: '1',
    query: 'Тестовый поиск',
    tenders: mockTenders
  }]);
  const [activeKey, setActiveKey] = useState<string>('1');
  const [selectedTender, setSelectedTender] = useState<string | null>(null);

  const handleSearch = async (query: string) => {
    // В реальном приложении здесь будет API запрос
    const newTab: Tab = {
      id: Date.now().toString(),
      query,
      tenders: mockTenders // Используем моковые данные
    };
    
    setTabs([...tabs, newTab]);
    setActiveKey(newTab.id);
  };

  const formatAmount = (amount: string) => {
    return new Intl.NumberFormat('ru-RU', {
      style: 'currency',
      currency: 'RUB',
      maximumFractionDigits: 0
    }).format(Number(amount));
  };

  return (
    <div className="app">
      <div className="search-container">
        <Input.Search
          placeholder="Введите поисковый запрос..."
          enterButton={<SearchOutlined />}
          size="large"
          onSearch={handleSearch}
        />
      </div>
      
      <Tabs
        activeKey={activeKey}
        onChange={setActiveKey}
        type="editable-card"
        items={tabs.map(tab => ({
          key: tab.id,
          label: tab.query,
          children: (
            <List
              grid={{ gutter: 16, column: 1 }}
              dataSource={tab.tenders}
              renderItem={tender => (
                <List.Item>
                  <Card 
                    onClick={() => setSelectedTender(tender.purchase_number)}
                    style={{ cursor: 'pointer' }}
                    hoverable
                  >
                    <div style={{ marginBottom: '16px' }}>
                      <Tag color="blue">{tender.purchase_number}</Tag>
                    </div>
                    
                    <Text strong style={{ fontSize: '16px', display: 'block', marginBottom: '12px' }}>
                      {tender.tender_name}
                    </Text>
                    
                    <div style={{ display: 'flex', gap: '24px', color: '#666' }}>
                      {tender.deadline && (
                        <Text>
                          <CalendarOutlined style={{ marginRight: '8px' }} />
                          {new Date(tender.deadline).toLocaleDateString('ru-RU')}
                        </Text>
                      )}
                      {tender.amount && (
                        <Text>
                          <DollarOutlined style={{ marginRight: '8px' }} />
                          {formatAmount(tender.amount)}
                        </Text>
                      )}
                      {tender.organization && (
                        <Text>
                          <BankOutlined style={{ marginRight: '8px' }} />
                          {tender.organization}
                        </Text>
                      )}
                    </div>
                  </Card>
                </List.Item>
              )}
            />
          )
        }))}
      />
      
      {selectedTender && (
        <TenderDetails
          purchaseNumber={selectedTender}
          visible={!!selectedTender}
          onClose={() => setSelectedTender(null)}
        />
      )}
    </div>
  );
};

export default App; 