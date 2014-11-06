<?php

class Tags extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'tags';
	}

	public function rules()
	{
		return array(
			array('descripcion', 'required'),
			array('documento', 'numerical', 'integerOnly'=>true),
			array('descripcion', 'length', 'max'=>128),
			array('id, descripcion, documento', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'documento0' => array(self::BELONGS_TO, 'Documentos', 'documento'),
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
			'descripcion' => Yii::t('app', 'Descripcion'),
			'documento' => Yii::t('app', 'Documento'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('descripcion',$this->descripcion,true);

		$criteria->compare('documento',$this->documento);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
