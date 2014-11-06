<?php

class TablasDeContenido extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'tablasDeContenido';
	}

	public function rules()
	{
		return array(
			array('indice, descripcion', 'required'),
			array('metaDoc', 'numerical', 'integerOnly'=>true),
			array('indice', 'length', 'max'=>64),
			array('descripcion', 'length', 'max'=>256),
			array('id, indice, descripcion, metaDoc', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
			'metaDoc0' => array(self::BELONGS_TO, 'MetaDocs', 'metaDoc'),
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
			'indice' => Yii::t('app', 'Indice'),
			'descripcion' => Yii::t('app', 'Descripcion'),
			'metaDoc' => Yii::t('app', 'Meta Doc'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('indice',$this->indice,true);

		$criteria->compare('descripcion',$this->descripcion,true);

		$criteria->compare('metaDoc',$this->metaDoc);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}
