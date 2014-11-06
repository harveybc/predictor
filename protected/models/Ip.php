<?php

class Ip extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'ip';
	}

	public function rules()
	{
		return array(
			array('IP', 'numerical'),
			array('TAG', 'length', 'max'=>50),
			array('Fecha', 'safe'),
			array('id, Fecha, TAG, IP', 'safe', 'on'=>'search'),
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
			'TAG' => Yii::t('app', 'Tag'),
			'IP' => Yii::t('app', 'Ip'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('Fecha',$this->Fecha,true);

		$criteria->compare('TAG',$this->TAG,true);

		$criteria->compare('IP',$this->IP);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
