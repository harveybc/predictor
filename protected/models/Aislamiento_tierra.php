<?php

class Aislamiento_tierra extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'aislamiento_tierra';
	}

	public function rules()
	{
		return array(
			array('TAG, Fecha', 'required'),
			array('Toma, OT', 'numerical', 'integerOnly'=>true),
			array('A025, A050, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, B025, B050, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, C025, C050, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10', 'numerical'),
                        array('Estado', 'numerical', 'integerOnly'=>true),
                        array('Observaciones', 'length', 'max'=>250),
			array('TAG', 'length', 'max'=>50),
			array('OT, Estado, Observaciones,Toma, TAG, Fecha, A025, A050, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, B025, B050, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, C025, C050, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, OT', 'safe', 'on'=>'search'),
                        array('A1,B1,C1', 'compare', 'compareValue'=>0,'operator'=>'!=','message'=>'Este valor no puede ser 0'),                   
                    
		);
	}

	public function relations()
	{
		return array(
                    
                    'relacMotores' => array(self::BELONGS_TO, 'Motores', '','on'=>'TAG=TAG'),
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
			'Toma' => Yii::t('app', 'Toma'),
			'TAG' => Yii::t('app', 'Motor'),
			'Fecha' => Yii::t('app', 'Fecha'),
			'A025' => Yii::t('app', 'A025'),
			'A050' => Yii::t('app', 'A050'),
			'A1' => Yii::t('app', 'A1'),
			'A2' => Yii::t('app', 'A2'),
			'A3' => Yii::t('app', 'A3'),
			'A4' => Yii::t('app', 'A4'),
			'A5' => Yii::t('app', 'A5'),
			'A6' => Yii::t('app', 'A6'),
			'A7' => Yii::t('app', 'A7'),
			'A8' => Yii::t('app', 'A8'),
			'A9' => Yii::t('app', 'A9'),
			'A10' => Yii::t('app', 'A10'),
			'B025' => Yii::t('app', 'B025'),
			'B050' => Yii::t('app', 'B050'),
			'B1' => Yii::t('app', 'B1'),
			'B2' => Yii::t('app', 'B2'),
			'B3' => Yii::t('app', 'B3'),
			'B4' => Yii::t('app', 'B4'),
			'B5' => Yii::t('app', 'B5'),
			'B6' => Yii::t('app', 'B6'),
			'B7' => Yii::t('app', 'B7'),
			'B8' => Yii::t('app', 'B8'),
			'B9' => Yii::t('app', 'B9'),
			'B10' => Yii::t('app', 'B10'),
			'C025' => Yii::t('app', 'C025'),
			'C050' => Yii::t('app', 'C050'),
			'C1' => Yii::t('app', 'C1'),
			'C2' => Yii::t('app', 'C2'),
			'C3' => Yii::t('app', 'C3'),
			'C4' => Yii::t('app', 'C4'),
			'C5' => Yii::t('app', 'C5'),
			'C6' => Yii::t('app', 'C6'),
			'C7' => Yii::t('app', 'C7'),
			'C8' => Yii::t('app', 'C8'),
			'C9' => Yii::t('app', 'C9'),
			'C10' => Yii::t('app', 'C10'),
			'OT' => Yii::t('app', 'Orden de Trabajo'),
                        'Estado' => Yii::t('app', 'Estado'),
                        'Observaciones' => Yii::t('app', 'Observaciones'),            
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;


//		$criteria->compare('Toma',$this->Toma,true);

		//$criteria->compare('TAG',$this->TAG,true);
        

	//	$criteria->compare('Fecha',$this->Fecha,true);
        $criteria->compare('OT',$this->OT);

        $criteria->compare('Estado', $this->Estado);
/*
		$criteria->compare('A025',$this->A025);

		$criteria->compare('A050',$this->A050);

		$criteria->compare('A1',$this->A1);

		$criteria->compare('A2',$this->A2);

		$criteria->compare('A3',$this->A3);

		$criteria->compare('A4',$this->A4);

		$criteria->compare('A5',$this->A5);

		$criteria->compare('A6',$this->A6);

		$criteria->compare('A7',$this->A7);

		$criteria->compare('A8',$this->A8);

		$criteria->compare('A9',$this->A9);

		$criteria->compare('A10',$this->A10);

		$criteria->compare('B025',$this->B025);

		$criteria->compare('B050',$this->B050);

		$criteria->compare('B1',$this->B1);

		$criteria->compare('B2',$this->B2);

		$criteria->compare('B3',$this->B3);

		$criteria->compare('B4',$this->B4);

		$criteria->compare('B5',$this->B5);

		$criteria->compare('B6',$this->B6);

		$criteria->compare('B7',$this->B7);

		$criteria->compare('B8',$this->B8);

		$criteria->compare('B9',$this->B9);

		$criteria->compare('B10',$this->B10);

		$criteria->compare('C025',$this->C025);

		$criteria->compare('C050',$this->C050);

		$criteria->compare('C1',$this->C1);

		$criteria->compare('C2',$this->C2);

		$criteria->compare('C3',$this->C3);

		$criteria->compare('C4',$this->C4);

		$criteria->compare('C5',$this->C5);

		$criteria->compare('C6',$this->C6);

		$criteria->compare('C7',$this->C7);

		$criteria->compare('C8',$this->C8);

		$criteria->compare('C9',$this->C9);

		$criteria->compare('C10',$this->C10);

		
*/
		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
    
    public function searchFechas($TAG)
        {
            if ($TAG=="")
                $TAG="ZZ_ZZ_Z";
                        $criteria = new CDbCriteria;
//            $criteria->compare('TAG',
//                               $TAG,
//                               false);
            return new CActiveDataProvider(get_class($this), array(
                    //'criteria' => $criteria,
                    'criteria' => array('order'=>'Fecha DESC','condition'=>'TAG="'.$TAG.'"'),
                ));
  /*          return new CActiveDataProvider(get_class($this), array (
                'criteria' => $criteria,
            ));
            
            $sql = 'SELECT * FROM motores WHERE Equipo="' . $equipo . '" AND Area="'.$area.'" ORDER BY Equipo';
            $myDataProvider = new CSqlDataProvider($sql, array (
                    'keyField' => 'id',
                    'pagination' => array (
                        'pageSize' => 10,
                    ),
                    )
            );
            return $myDataProvider;
   * 
   */
             
             
        }

}
