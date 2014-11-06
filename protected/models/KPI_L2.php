<?php

class KPI_L2 extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'KPI_L2';
	}

	public function rules()
	{
		return array(
			array('Fecha, Eff_Shift, Count_Fill_Batch, Count_Pall_Batch, Count_Fill_Shift', 'required'),
			array('Eff_Shift, Count_Fill_Batch, Count_Pall_Batch, Count_Fill_Shift', 'numerical', 'integerOnly'=>true),
			array('id, Fecha, Eff_Shift, Count_Fill_Batch, Count_Pall_Batch, Count_Fill_Shift', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
		);
	}

	public function behaviors()
	{
		return array('CAdvancedArBehavior',
				array('class' => 'ext.CAdvancedArBehavior')
				);
	}

	public function attributeLabels()
	{
		return array(
			'id' => Yii::t('app', 'ID'),
			'Fecha' => Yii::t('app', 'Fecha'),
			'Eff_Shift' => Yii::t('app', 'Eff Shift'),
			'Count_Fill_Batch' => Yii::t('app', 'Count Fill Batch'),
			'Count_Pall_Batch' => Yii::t('app', 'Count Pall Batch'),
			'Count_Fill_Shift' => Yii::t('app', 'Count Fill Shift'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('Fecha',$this->Fecha,true);

		$criteria->compare('Eff_Shift',$this->Eff_Shift);

		$criteria->compare('Count_Fill_Batch',$this->Count_Fill_Batch);

		$criteria->compare('Count_Pall_Batch',$this->Count_Pall_Batch);

		$criteria->compare('Count_Fill_Shift',$this->Count_Fill_Shift);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
