import React, { useEffect, useState } from 'react';
import { Modal, Descriptions, Spin } from 'antd';

interface TenderDetailsProps {
  purchaseNumber: string;
  visible: boolean;
  onClose: () => void;
}

interface TenderDetail {
  purchase_number: string;
  tender_name: string;
  description?: string;
  organization?: string;
  amount?: number;
  deadline?: string;
  contact_info?: {
    name?: string;
    email?: string;
    phone?: string;
  };
}

const TenderDetails: React.FC<TenderDetailsProps> = ({ 
  purchaseNumber, 
  visible, 
  onClose 
}) => {
  const [details, setDetails] = useState<TenderDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchDetails = async () => {
      if (!visible) return;
      
      setLoading(true);
      try {
        const response = await fetch(
          `http://localhost:8000/tender/${purchaseNumber}`
        );
        const data = await response.json();
        setDetails(data);
      } catch (error) {
        console.error('Error fetching tender details:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDetails();
  }, [purchaseNumber, visible]);

  return (
    <Modal
      title="Детальная информация о тендере"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={800}
    >
      {loading ? (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Spin size="large" />
        </div>
      ) : details ? (
        <Descriptions bordered column={1}>
          <Descriptions.Item label="Номер закупки">
            {details.purchase_number}
          </Descriptions.Item>
          <Descriptions.Item label="Наименование">
            {details.tender_name}
          </Descriptions.Item>
          {details.description && (
            <Descriptions.Item label="Описание">
              {details.description}
            </Descriptions.Item>
          )}
          {details.organization && (
            <Descriptions.Item label="Организация">
              {details.organization}
            </Descriptions.Item>
          )}
          {details.amount && (
            <Descriptions.Item label="Сумма">
              {details.amount.toLocaleString('ru-RU')} ₽
            </Descriptions.Item>
          )}
          {details.deadline && (
            <Descriptions.Item label="Срок подачи">
              {new Date(details.deadline).toLocaleDateString('ru-RU')}
            </Descriptions.Item>
          )}
          {details.contact_info && (
            <Descriptions.Item label="Контактная информация">
              {details.contact_info.name && <div>ФИО: {details.contact_info.name}</div>}
              {details.contact_info.email && <div>Email: {details.contact_info.email}</div>}
              {details.contact_info.phone && <div>Телефон: {details.contact_info.phone}</div>}
            </Descriptions.Item>
          )}
        </Descriptions>
      ) : (
        <div>Информация не найдена</div>
      )}
    </Modal>
  );
};

export default TenderDetails; 