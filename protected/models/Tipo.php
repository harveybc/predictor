<?php

class Tipo extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'tipo';
	}

	public function rules()
	{
		return array(
			array('Tipo_Aceite, Proceso', 'length', 'max'=>50),
			array('id, Tipo_Aceite, Proceso', 'safe', 'on'=>'search'),
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
			'Tipo_Aceite' => Yii::t('app', 'Tipo Aceite'),
			'Proceso' => Yii::t('app', 'Area'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('Tipo_Aceite',$this->Tipo_Aceite,true);

		$criteria->compare('Proceso',$this->Proceso,true);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
                    
		));
	}
        
         public function searchFechas($area)
        {
            if ($area=="")
                $area="ZZ_ZZ_Z";
                        $criteria = new CDbCriteria;
            $criteria->compare('Proceso',
                               $area,
                               true);
            return new CActiveDataProvider(get_class($this), array (
                'criteria' => $criteria,
            ));
        }
}
